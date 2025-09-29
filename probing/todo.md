# To Do's / Questions

- have it use a train/val/test split instead of train/test?
- right now it's only saving the best model per sweep, maybe we want all of them?
- functionality to train all layers in a single 'run' once we settle on hyperparams
- once we train all layers, plot probe accuracy by layer (later layers should have higher accuracy I'd assume)
- plot probe accuracy by sequence length? can access sequence length by layer
- do all three types of probing (already set up in the custom torch dataset class)
- wandb logging if we're training for many epochs, not sure how necessary this is
- classes aren't balanced right now across train/test or in general (could be biased towards answers if responses are longer for some questions)
- get a baseline by assigning random labels and comparing a probe's accuracy on that to our actual dataset