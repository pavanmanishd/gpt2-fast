use std::cell::Cell;
use std::marker::PhantomData;
use std::ops::Range;

use crate::Error;

// Waking a parked worker costs more than the gap between the small parallel
// regions inside one token, so workers spin briefly before parking.
const SPIN_ROUNDS: usize = 4096;

pub(crate) struct Threads {
    pool: Option<Pool>,
    // `run` publishes a job into shared state, so only one thread may enter it.
    _single_runner: PhantomData<Cell<()>>,
}

impl Threads {
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
    pub(crate) fn from_env() -> Result<Threads, Error> {
        let count = match std::env::var("GPT2_THREADS") {
            Ok(value) => value
                .parse::<usize>()
                .map_err(|_| Error::new("GPT2_THREADS must be a positive integer"))?,
            Err(_) => std::thread::available_parallelism().map_or(1, |count| count.get()),
        };
        Threads::with_count(count)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    pub(crate) fn from_env() -> Result<Threads, Error> {
        // The count arrives from JS after the page reads its own query string.
        Threads::with_count(1)
    }

    pub(crate) fn with_count(count: usize) -> Result<Threads, Error> {
        if count == 0 {
            return Err(Error::new("thread count must be at least 1"));
        }
        let pool = if count == 1 {
            None
        } else {
            Some(Pool::new(count)?)
        };
        Ok(Threads {
            pool,
            _single_runner: PhantomData,
        })
    }

    pub(crate) fn run(
        &self,
        output: &mut [f32],
        align: usize,
        task: impl Fn(usize, &mut [f32]) + Sync,
    ) {
        match &self.pool {
            Some(pool) => pool.run(output, align, &task),
            None => task(0, output),
        }
    }
}

struct Job {
    task: &'static (dyn Fn(usize, &mut [f32]) + Sync),
    parts: Vec<(usize, *mut [f32])>,
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
mod backend {
    use std::cell::UnsafeCell;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread::{self, JoinHandle};

    use super::{boundary, bounds, Job, SPIN_ROUNDS};
    use crate::Error;

    pub(super) struct Pool {
        shared: Arc<Shared>,
        workers: Vec<JoinHandle<()>>,
    }

    struct Shared {
        generation: AtomicU64,
        completed: AtomicUsize,
        shutdown: AtomicBool,
        job: UnsafeCell<Option<Job>>,
    }

    // SAFETY: the job slot is written by the calling thread before it publishes the
    // generation and after every worker has finished the previous job. A worker reads
    // the slot only after it observes that generation, and each part is handed to
    // exactly one worker.
    unsafe impl Send for Shared {}
    unsafe impl Sync for Shared {}

    impl Pool {
        pub(super) fn new(count: usize) -> Result<Pool, Error> {
            let shared = Arc::new(Shared {
                generation: AtomicU64::new(0),
                completed: AtomicUsize::new(0),
                shutdown: AtomicBool::new(false),
                job: UnsafeCell::new(None),
            });
            let mut workers = Vec::with_capacity(count - 1);
            for index in 0..count - 1 {
                let worker_shared = Arc::clone(&shared);
                let spawned = thread::Builder::new()
                    .name(format!("gpt2-{index}"))
                    .spawn(move || worker(worker_shared, index));
                match spawned {
                    Ok(worker) => workers.push(worker),
                    Err(error) => {
                        stop(&shared, workers);
                        return Err(Error::new(format!("cannot spawn worker thread: {error}")));
                    }
                }
            }
            Ok(Pool { shared, workers })
        }

        pub(super) fn run(
            &self,
            output: &mut [f32],
            align: usize,
            task: &(dyn Fn(usize, &mut [f32]) + Sync),
        ) {
            let count = self.workers.len() + 1;
            let len = output.len();
            let (mine, mut rest) = output.split_at_mut(boundary(1, len, count, align));

            // SAFETY: every worker finishes before `run` returns, so the task reference
            // and the output slices outlive every worker's use of them.
            let task: &'static (dyn Fn(usize, &mut [f32]) + Sync) =
                unsafe { std::mem::transmute(task) };
            // SAFETY: no worker is reading the slot; the previous job completed.
            let job = unsafe {
                (*self.shared.job.get()).get_or_insert_with(|| Job {
                    task,
                    parts: Vec::with_capacity(count - 1),
                })
            };
            job.task = task;
            job.parts.clear();
            for index in 1..count {
                let range = bounds(index, len, count, align);
                let (part, tail) = rest.split_at_mut(range.len());
                job.parts.push((range.start, part as *mut [f32]));
                rest = tail;
            }

            self.shared.completed.store(0, Ordering::Relaxed);
            self.shared.generation.fetch_add(1, Ordering::Release);
            for worker in &self.workers {
                worker.thread().unpark();
            }

            task(0, mine);
            while self.shared.completed.load(Ordering::Acquire) < count - 1 {
                std::hint::spin_loop();
            }
        }
    }

    impl Drop for Pool {
        fn drop(&mut self) {
            stop(&self.shared, std::mem::take(&mut self.workers));
        }
    }

    fn stop(shared: &Shared, workers: Vec<JoinHandle<()>>) {
        shared.shutdown.store(true, Ordering::Release);
        shared.generation.fetch_add(1, Ordering::Release);
        for worker in &workers {
            worker.thread().unpark();
        }
        for worker in workers {
            let _ = worker.join();
        }
    }

    fn worker(shared: Arc<Shared>, index: usize) {
        let mut seen = 0;
        loop {
            let mut spins = 0;
            let generation = loop {
                let generation = shared.generation.load(Ordering::Acquire);
                if generation != seen {
                    break generation;
                }
                if spins < SPIN_ROUNDS {
                    std::hint::spin_loop();
                    spins += 1;
                } else {
                    thread::park();
                    spins = 0;
                }
            };
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            seen = generation;

            // SAFETY: the caller publishes the job before the generation and keeps it
            // alive until this worker reports completion below.
            let job = unsafe {
                (*shared.job.get())
                    .as_ref()
                    .expect("job follows generation")
            };
            let (offset, part) = job.parts[index];
            // SAFETY: each part belongs to exactly one worker.
            (job.task)(offset, unsafe { &mut *part });
            shared.completed.fetch_add(1, Ordering::Release);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
mod backend {
    use std::arch::wasm32::{memory_atomic_notify, memory_atomic_wait32};
    use std::cell::UnsafeCell;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    use super::{boundary, bounds, Job, SPIN_ROUNDS};
    use crate::Error;

    static POOL: AtomicUsize = AtomicUsize::new(0);

    pub(super) struct Pool {
        shared: *mut Shared,
        count: usize,
    }

    struct Shared {
        generation: AtomicU32,
        completed: AtomicU32,
        shutdown: AtomicU32,
        count: u32,
        job: UnsafeCell<Option<Job>>,
    }

    // SAFETY: the job slot is written by the calling thread before it publishes the
    // generation and after every worker has finished the previous job. A worker reads
    // the slot only after it observes that generation, and each part is handed to
    // exactly one worker.
    unsafe impl Send for Shared {}
    unsafe impl Sync for Shared {}

    impl Pool {
        pub(super) fn new(count: usize) -> Result<Pool, Error> {
            let shared = Box::into_raw(Box::new(Shared {
                generation: AtomicU32::new(0),
                completed: AtomicU32::new(0),
                shutdown: AtomicU32::new(0),
                count: count as u32,
                job: UnsafeCell::new(None),
            }));
            POOL.store(shared as usize, Ordering::Release);
            Ok(Pool { shared, count })
        }

        pub(super) fn run(
            &self,
            output: &mut [f32],
            align: usize,
            task: &(dyn Fn(usize, &mut [f32]) + Sync),
        ) {
            let count = self.count;
            let len = output.len();
            let (mine, mut rest) = output.split_at_mut(boundary(1, len, count, align));

            // SAFETY: every worker finishes before `run` returns, so the task reference
            // and the output slices outlive every worker's use of them.
            let task: &'static (dyn Fn(usize, &mut [f32]) + Sync) =
                unsafe { std::mem::transmute(task) };
            // SAFETY: this instance is the only publisher, and the previous job completed.
            let shared = unsafe { &*self.shared };
            // SAFETY: no worker is reading the slot; the previous job completed.
            let job = unsafe {
                (*shared.job.get()).get_or_insert_with(|| Job {
                    task,
                    parts: Vec::with_capacity(count - 1),
                })
            };
            job.task = task;
            job.parts.clear();
            for index in 1..count {
                let range = bounds(index, len, count, align);
                let (part, tail) = rest.split_at_mut(range.len());
                job.parts.push((range.start, part as *mut [f32]));
                rest = tail;
            }

            shared.completed.store(0, Ordering::Relaxed);
            shared.generation.fetch_add(1, Ordering::Release);
            // SAFETY: workers park on the generation word; a surplus wake is harmless.
            unsafe {
                memory_atomic_notify(shared.generation.as_ptr().cast(), count as u32);
            }

            task(0, mine);
            let mut done = shared.completed.load(Ordering::Acquire);
            while done < count as u32 - 1 {
                // SAFETY: this runs in a Web Worker, the only place a browser allows blocking.
                unsafe {
                    memory_atomic_wait32(shared.completed.as_ptr().cast(), done as i32, -1);
                }
                done = shared.completed.load(Ordering::Acquire);
            }
        }
    }

    impl Drop for Pool {
        fn drop(&mut self) {
            // SAFETY: the block is leaked below, so the pointer stays valid even if a
            // worker is still on its way to observe the shutdown flag.
            let shared = unsafe { &*self.shared };
            shared.shutdown.store(1, Ordering::Release);
            shared.generation.fetch_add(1, Ordering::Release);
            // SAFETY: workers park on the generation word.
            unsafe {
                memory_atomic_notify(shared.generation.as_ptr().cast(), self.count as u32);
            }
            POOL.compare_exchange(self.shared as usize, 0, Ordering::AcqRel, Ordering::Acquire)
                .ok();
        }
    }

    pub(super) fn pointer() -> usize {
        POOL.load(Ordering::Acquire)
    }

    pub(super) fn worker_count() -> usize {
        let pointer = pointer();
        if pointer == 0 {
            0
        } else {
            // SAFETY: the pointer is published by `Pool::new` and never freed.
            unsafe { (*(pointer as *const Shared)).count as usize - 1 }
        }
    }

    pub(super) fn worker_main(pointer: usize, index: usize) {
        // SAFETY: the pointer comes from `Pool::new` and the block is never freed.
        let shared = unsafe { &*(pointer as *const Shared) };
        let mut seen = 0;
        loop {
            let mut spins = 0;
            let generation = loop {
                let generation = shared.generation.load(Ordering::Acquire);
                if generation != seen {
                    break generation;
                }
                if spins < SPIN_ROUNDS {
                    std::hint::spin_loop();
                    spins += 1;
                } else {
                    // SAFETY: this runs in a Web Worker, where blocking is allowed.
                    unsafe {
                        memory_atomic_wait32(shared.generation.as_ptr().cast(), seen as i32, -1);
                    }
                    spins = 0;
                }
            };
            if shared.shutdown.load(Ordering::Acquire) != 0 {
                return;
            }
            seen = generation;

            // SAFETY: the caller publishes the job before the generation and keeps it
            // alive until this worker reports completion below.
            let job = unsafe {
                (*shared.job.get())
                    .as_ref()
                    .expect("job follows generation")
            };
            let (offset, part) = job.parts[index];
            // SAFETY: each part belongs to exactly one worker.
            (job.task)(offset, unsafe { &mut *part });
            shared.completed.fetch_add(1, Ordering::Release);
            // SAFETY: the caller parks on the completion word.
            unsafe {
                memory_atomic_notify(shared.completed.as_ptr().cast(), 1);
            }
        }
    }
}

use backend::Pool;

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub fn thread_pool_pointer() -> usize {
    backend::pointer()
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub fn thread_pool_workers() -> usize {
    backend::worker_count()
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub fn thread_pool_worker_main(pointer: usize, index: usize) {
    backend::worker_main(pointer, index);
}

fn bounds(index: usize, len: usize, count: usize, align: usize) -> Range<usize> {
    boundary(index, len, count, align)..boundary(index + 1, len, count, align)
}

fn boundary(index: usize, len: usize, count: usize, align: usize) -> usize {
    if index == count {
        len
    } else {
        (index * len / count) / align * align
    }
}
