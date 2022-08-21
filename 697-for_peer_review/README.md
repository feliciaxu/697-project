# 697
run the review.ipynb file for the temp demo. It is just a task of the class.
### Our dataset is EUVP,UIEB and mix dataset
We can get offical EUVP dataset from https://irvlab.cs.umn.edu/resources/euvp-dataset


We can get offical UIEB dataset from https://li-chongyi.github.io/proj_benchmark.html


The mix dataset can be found in this project



### The role of each folder
Evaluation folder contains the evaluation function, PSNR SSIM and UIQM


Gui folder contains the GUI.py, we can use it to interactive with others


model contains the trained model


We can found some test results in the test_result folder. We use EUVP offical test dataset for 9 models( Two different versions of the three subsets, two different versions of the mix dataset, and unpaired）

### Python files
We can run the pair_train.py to train the paired data, unpair_train.py to train the unpaired data, maintest.py to test to pre-train model

### Sample image
![图片](https://user-images.githubusercontent.com/45815690/185768844-70714f78-8686-41fe-8ebf-6949fcbd8e68.png)
The left is the origin one. The right is after using dark model. 

### Conclusion
We designed two versions of the paired model, which use a five- and six-layer U-Net structure for the generator and PatchGAN for the discriminator part that can check the quality of the generated images. The loss functions of Pair and Unpair are also different. The model evaluates the pictures' global content, color, local texture, and style using three separate assessment criteria: PSNR, SSIM, and UIQM. Finally, we use GUI to show the output image. 


The performance of the model corresponding to each of the three paired subsets is generally somewhat better than that of the other two subsets, according to our analysis of the data, and the performance of the mixed dataset is better than that of the other three subsets. The model in version 2 performs better overall than version 1, but training takes longer. The UIQM of the mixed dataset, on the other hand, is not much better and may even be worse than the other subsets, according to the comparison. This is most likely because I still need to alter the parameter weights for UIQM, which I will do as my next task. In addition, it may be necessary to consider a more objective criterion for selection rather than random sampling for mixed datasets. We can find more details in my report. 
