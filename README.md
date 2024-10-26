# 335_Model

The overall structure of the code remains similar to the original, but with these efficiency improvements and bug fixes. You can now use this `model2.py` file for your translation model training.

1. Fixed typos and syntax errors in the `TranslationDataset` class.
2. Used vectorized operations for calculating max lengths in the `get_ds` function.
3. Implemented multiprocessing for data loading with `num_workers=4` and `pin_memory=True`.
4. Added mixed precision training using `torch.cuda.amp`.
5. Implemented a learning rate scheduler using `OneCycleLR`.
6. Used `torch.jit.script` for potential performance improvements.
