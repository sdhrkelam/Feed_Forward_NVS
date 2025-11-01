class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 image_num_range,
                 h_range,
                 epoch=0,
                 seed=42,
                 max_img_per_gpu=48,
                 use_fixed_indices=False):  # Add this parameter
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            image_num_range: List containing [min_images, max_images] per sample.
            h_range: Height range for patches.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
            use_fixed_indices: If True, use fixed num_context_views instead of random sampling.
        """
        self.sampler = sampler
        self.image_num_range = image_num_range
        self.h_range = h_range
        self.rng = random.Random()
        self.use_fixed_indices = use_fixed_indices
        
        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: float(num_images**2) for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])
        
        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number or use fixed value
                if self.use_fixed_indices:
                    # Use the maximum value from image_num_range (num_context_views)
                    random_image_num = self.image_num_range[1]
                else:
                    random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                
                random_ps_h = np.random.randint(low=(self.h_range[0] // 14), high=(self.h_range[1] // 14)+1)

                # Update sampler parameters
                self.sampler.update_parameters(
                    image_num=random_image_num,
                    ps_h=random_ps_h
                )
                
                # Calculate batch size based on max images per GPU and current image number
                batch_size = self.max_img_per_gpu / random_image_num
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)  # Ensure batch size is at least 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, aspect_ratio, image_num)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Return a large dummy length
        return 1000000



class MixedBatchSampler(BatchSampler):
    """Sample one batch from a selected dataset with given probability.
    Compatible with datasets at different resolution
    """

    def __init__(
        self, src_dataset_ls, batch_size, num_context_views, world_size=1, rank=0, prob=None, sampler=None, generator=None
    ):
        self.base_sampler = None
        self.batch_size = batch_size
        self.num_context_views = num_context_views
        self.world_size = world_size
        self.rank = rank
        self.drop_last = True
        self.generator = generator

        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)
        
        # Dataset length
        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]  # cumulative dataset length
        
        # BatchSamplers for each source dataset
        self.src_batch_samplers = []
        for ds in self.src_dataset_ls:
            sampler = DynamicDistributedSampler(ds, num_replicas=self.world_size, rank=self.rank, seed=42, shuffle=True)
            sampler.set_epoch(0)

            if hasattr(ds, "epoch"):
                ds.epoch = 0
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(0)
            
            # Check if dataset has use_fixed_indices config
            use_fixed_indices = getattr(ds.cfg.view_sampler, 'use_fixed_indices', False)
            
            batch_sampler = DynamicBatchSampler(
                sampler, 
                [2, ds.cfg.view_sampler.num_context_views], 
                ds.cfg.input_image_shape,
                seed=42,
                max_img_per_gpu=ds.cfg.view_sampler.max_img_per_gpu,
                use_fixed_indices=use_fixed_indices  # Pass the flag here
            )
            self.src_batch_samplers.append(batch_sampler)
        
        print("Setting epoch for all underlying BatchedRandomSamplers")
        self.raw_batches = [
            list(bs) for bs in self.src_batch_samplers
        ]  # index in original dataset
        self.n_batches = [len(b) for b in self.raw_batches]
        self.n_total_batch = sum(self.n_batches)
        
        # sampling probability
        if prob is None:
            # if not given, decide by dataset length
            self.prob = torch.tensor(self.n_batches) / self.n_total_batch
        else:
            self.prob = torch.as_tensor(prob)
    
    # ...existing code...