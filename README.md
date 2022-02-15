# A-Study-on-Patch-Wise-Deepfake-Image-Detection

![image](https://user-images.githubusercontent.com/77098071/152724878-2de1fe81-b425-438d-8dff-a4960acee09c.png)

__IVC(Image & Vision Computing) Lab / Pukyong Nat'l Univ Electronic Engineering / Busan, Republic of Korea__   
Jeonghan Lee, Hanhoon Park(Major Professor)

* Paper(Korean) : "A Study on Patch-Wise Deepfake Image Detection"  (*Attach the pdf file*)   

Abstract : In this paper, to detect deepfake images, images are divided into patches, determined whether they are forged for each patch, and the discrimination results for each patch are presented, and the detection preformance of images generated with StyleGAN2 is verified through experiments. As a result of the experiment, it was confirmed that the detection accuracy varies depending on the patch size, but overall, the patch-based method can greatly improve detection accuracy and further improve detection accuracy by adding a process of selecting highly reliable patches.

## Settings
### Dataset
* Category : FFHQ, CelebA, LSUN - church_outdoor
* Real/Fake 각각 Train set 25K장 / Test set 2K장 임의추출하여 사용 -> 각 카테고리(FFHQ, CelebA, LSUN)당 54K장의 이미지 사용
* Image size : 256X256 (pixel size)
* Patch size : 128X128, 64X64, 32X32, 16X16
* GAN : SyleGAN2, ProGAN

### Feature Extraction
* Xception

### Spec
* Loss fuction = Cross Entropy, epoch = 50, optimizer = Adam
* Indicators : The highest accuracy(top-1) / The average of the 5 highest accuracy(top-5) / The average of the overall accuracy(Average)

## Experimental Method
### Experiment 1
To prevent overlapping images, patches divided into various sizes are learned separately, and classification accuracy is calculated after evaluating whether the validation
dataset is forged by considering each patch as a single image.
![image](https://user-images.githubusercontent.com/77098071/152726987-bceb617f-2152-4211-8b73-63a63c6f9c4d.png)
### Experiment 2
Identify fake images for patches that do not overlap inside the input image, and determine wheter a single in put image(4K images) is forged from the number of patches determined. (*Set to identify fake if the results in the single image are tied*)
![image](https://user-images.githubusercontent.com/77098071/152727459-5133263e-a68f-4a88-ae42-68bdf2ca9e13.png)
### Experiment 3
Identify fake images for randomly-cropped patch inside the input image and determine whether a single input image(4K images) is forged from the number of determined patches.(*consisting of odd numbers to prevent a tie*)
![image](https://user-images.githubusercontent.com/77098071/152727820-55b251d1-1fcc-44af-8f60-77705f3981d7.png)
### Experiment 4
Determining whether an image is forged by selecting only patches with high classification reliability, excluding patches with low classification reliability without using all input patches. If all patches inside a single image are not applicable, **use all patches**.
![image](https://user-images.githubusercontent.com/77098071/152728977-029af5ea-6d8f-4407-8712-65592f02d55b.png)

## Result
### Experiment 1
![image](https://user-images.githubusercontent.com/77098071/152729058-6a41d55f-b897-4deb-9b90-a6bed42baa04.png) <br/>
**[Tab 1]** The accuracy of detecting fake patches according to the patch size
* As shown in **[Tab 1]**, the accuracy of fake patches detection varies depending on the patch size, but there is no significant difference
* When the patch size was 32X32, the accuracy was the highest
### Experiment 2
![image](https://user-images.githubusercontent.com/77098071/152729620-bcb88afc-acf0-43cc-8a39-b1d240116ce9.png) <br/>
**[Tab 2]** The accuracy of detecting of non-overlapping patch-wise fake images according to the patch size
* As shown in **[Tab 2]**, the smaller the patch size, the higher the detection accuarcy
* It can be seen that patch-wise fake image detection is valid
* When evaluated using non-overlapping patches, the accuracy is highest when the patch size is 32X32
### Experiment 3
![image](https://user-images.githubusercontent.com/77098071/152730461-fecb9cf3-08a4-490c-972e-6c1170db16fc.png) <br/>
**[Tab 3]** The accuracy of detecting randomly-cropped patch-wise fake images according to the patch size
* As shwon in **[Tab 3]**, similar to *Experiment 2*, the smaller the patch, the higher the detection accuracy
* However, unlike *Experiment 2*, when patches are obtained at any location, the full information cannot be utilized, so the detection accuracy is significantly reduced if the patch is not small enough (i.e., 64X64, 128X128).
### Experiment 4
![image](https://user-images.githubusercontent.com/77098071/152732572-b09186c8-07e5-4967-be92-a5d1329b4b29.png) <br/>
**[Tab 4]** The accuracy of detecting fake images using highly reliable patches
* As shown in **[Tab 4]**, *Experiment 4*is a threshold for selecting highly reliable patches, and after sorting the difference between the probability of real and fake for each patch according to the size, the top percentage of patches should be used
* However, the smaller theta, the more reliable patches are used, and if there is no patch satisfying the threshold value, all patches in the image are used to determine whether the image is foged or not, as shown in *Experiment 2*.(Therefore, the smaller theta, the more the results in **[Tab 4]** converge o nthe results in **[Tab 2]**
* *As a result*, accuracy is improved by selecting highly reliable patches and identifying fake images
### Experiment 5 - Comparative Experiment(Heterogeneous Validation Set)
![image](https://user-images.githubusercontent.com/77098071/152734822-a482c058-1547-42dd-8637-60f4e028c1b4.png) <br/>
**[Tab 5]** The accuracy of detecting fake images, when the content is different <br/><br/>
![image](https://user-images.githubusercontent.com/77098071/152734831-e1ef06cc-55ce-49f8-95e5-6998fefa5ac9.png) <br/>
**[Tab 6]** The accuracy of detecting fake images, when the generative model is different <br/>
* Assuming an validation dataset in an actual environment, if the conditions(generated content or generated model) are differnet, check the difference in classification performance that would have used the patch-processing
* When using the same training data, **[Tab 5]** shows the classification accuracy when evaluating the dataset that generated the chuch-outdoor image of the LSUN dataset using StyleGAN2, and **[Tab 6]** shows the classification accuracy when evaluating the dataset generated CelebA(face) images using ProGAN
* *As a result*, when patch-wise fake image detection is used, performance is greatly improved, indicating that it is valid to use validation datasets under different conditions

## Conclusion
* StyleGAN2로 생성된 **딥페이크 영상 검출**을 위해 입력 영상을 패치로 나누고, 패치의 인식 결과로부터 영상의 위조 여부를 판별하는 방법의 성능을 **실험적으로 검증**
* 패치의 크기에 따라 위조 영상 검출 정확도가 달라졌으나, **패치 기반 위조 영상 검출 방법은 유효**
* 신뢰도가 높은 패치만을 **선별함으로써** 검출 정확도를 보다 **개선**
* 딥러닝 기반 딥페이크 영상 검출 방법은 학습 데이터셋과 평가 데이터셋의 **생성 모델이 다르거나 콘텐츠가 상이**한 경우 검출 정확도가 크게 떨어지는데, 패치 기반 위조 영상 검출을 통해 **개선된 성능 확인**
* 향후, 더 어려운 조건에서도 **분류 정확도를 향상**시키기 위한 연구가 필요

### Acknowledgement
This work was supported by the National Re earch Foundation of Korea (NRF) Grant by the Korean Government through the MSIT under Grant 2021R1F1A1045749

### This content is inspired by the documents below:
1. T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” Proc. of CVPR, pp. 4401-4410, 2019.
2. Dlib, http://dlib.net/.
3. T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, “Analyzing and improving the image quality of StyleGAN,” Proc. of CVPR, pp. 8110-8119, 2020.
4. F. Chollet, “Xception: Deep learning with depthwis separable convolutions,” Proc. of CVPR, pp. 1251-1258, 2017
