# To Do's / Questions

- have it use a train/val/test split instead of train/test?
- functionality to train all layers in a single 'run' once we settle on hyperparams
- once we train all layers, plot probe accuracy by layer (later layers should have higher accuracy I'd assume)
- plot probe accuracy by sequence length? can access sequence length by layer
- do all three types of probing (already set up in the custom torch dataset class)
- wandb logging if we're training for many epochs, not sure how necessary this is
- get a baseline by assigning random labels and comparing a probe's accuracy on that to our actual dataset
- add random seeding