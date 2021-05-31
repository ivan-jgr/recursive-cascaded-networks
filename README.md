<span style="color:red"> This is a pytorch version of the [original tensorflow implementation](https://github.com/microsoft/Recursive-Cascaded-Networks) and for now only works with 2D images. </span>

# Recursive Cascaded Networks for Unsupervised Medical Image Registration

Paper link: [[arXiv]](https://arxiv.org/pdf/1907.12353)


*Recursive cascaded networks*, a general architecture that enables learning deep cascades, for deformable image registration. The moving image is warped successively by each cascade and finally aligned to the fixed image.

![cascade_example](./images/cascade_example.png)

![cascade_architecture](./images/cascade_architecture.png)

This repository includes:

* The recursive cascade network implementation with VTN as a base network for 2D images.

## Training

You need to define your own loader based on your dataset.

`python train.py -b BATCH_SIZE -n NUMBER_OF_CASCADES -e EPOCHS -i ITERS -iv VAL_ITERS -c FREQ_FOR_SAVE_MODEL -f SAMPLE_FOR VISUALIZATION`


## Acknowledgement

This is a pytorch version based on the official tensorflow implementation: https://github.com/microsoft/Recursive-Cascaded-Networks