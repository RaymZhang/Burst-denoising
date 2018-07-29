# Burst-denoising
Reproducing the results of https://arxiv.org/abs/1712.05790

OBSERVATIONS:
-During the first part of training time the loss from the single frame denoiser increase while the loss from the MFD decrease
-During the second part when the loss from the SFD and the MFD become quite near, both losses increase

-The trainburstserveur is faster than trainburstserveur2 when it comes to do one epoch of a small dataset, I still don't know why.

-There is still the drop of loss when there is a change of dataset

-I noticed that during testing time in testMFD the when showburst2 is launched twice, the second time the denoiser doesnt seems to work at all. It may be due to a lack of memory

PROBLEME:
The increase of the loss could be due to a probleme in memory management. When there is a change of dataset: maybe it refreshes the memory so that the denoiser 'is good again'.


TO DO:
 
Implement a version trainingfunction that concatenate one denoised burst then computing the loss burst wise (and not imagewise)

Launch the training for a few days with the new clean dataset to see if we achieve to get some interesting results.

Learn about how pytorch manage memory
