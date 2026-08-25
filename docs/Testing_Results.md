# Results of testing

### Few clarifications regarding test results:
> Original published images are available in [`images/`](images/).
> Processed test results are avaialble in [`result_images/`](result_images/).
> Images that followed the guideline, keeping all corners of A4 visible in image were tested.

> Most of images contain same person's hand, but there are exceptions: hand26, hand27 are different person's hand. One sample (hand23) is also of unknown person.
> Apart from above mentioned samples, images contained same person's hand and automatic A4 detection passed with all of those. (Including diff person's hand26 and hand27)

> Test samples had variety of different shots, including samples where reference object (A4 sheet) was tilted, camera angle tilted, different lightnings, shadows, background textures.


## Image results with automatic A4 detection 

> Table contains samples from same person's hand. Therefore, few images were left out of the table.
> Also few images e.g. hand4 - hand7 were left out due to A4 corners in images not visible.

> In table:
   > Test: column is for identifying which sample (image) was tested here. You can access these samples at /images
   > A4: whether automatic reference object (A4 sheet) was detected from sample
   > Hand: whether hand was detected from the perspective corrected image
   > Mask: whether handmask (silhouette) for hand was successfully drawn

| Test sample | A4 | Hand | Mask | Length | Palm width | Result |
|---|---|---|---|---:|---:|---|
| [sample1](images/hand1.jpg) | ✅ | ✅ | ✅ | 191.8 mm | 95.00 mm | [PASS](result_images/hand1_result.jpg) |
| [sample2](images/hand2.jpg) | ✅ | ✅ | ✅ | 183.1 mm | 91.7 mm | [PASS](result_images/hand2_result.jpg) |
| [sample3](images/hand8.jpg) | ✅ | ✅ | ✅ | 186.8 mm | 94.3 mm | [PASS](result_images/hand8_result.jpg) |
| [sample4](images/hand9.jpg) | ✅ | ✅ | ✅ | 188.3 mm | 93.00 mm | [PASS](result_images/hand9_result.jpg) |
| [sample5](images/hand10.jpg) | ✅ | ✅ | ✅ | 184.8 mm | 92.5 mm | [PASS](result_images/hand10_result.jpg) |
| [sample6](images/hand11.jpg) | ✅ | ✅ | ✅ | 187.2 mm | 91.8 mm | [PASS](result_images/hand11_result.jpg) |
| [sample7](images/hand12.jpg) | ✅ | ✅ | ✅ | 181.2 mm | 93.8 mm | [PASS](result_images/hand12_result.jpg) |
| [sample8](images/hand13.jpg) | ✅ | ✅ | ✅ | 184.7 mm | 92.8 mm | [PASS](result_images/hand13_result.jpg) |
| [sample9](images/hand3.jpg) | ✅ | ✅ | ✅ | 192.4 mm | 94.3 mm | [PASS](result_images/hand3_result.jpg) |
| [sample10](images/hand15.jpg) | ✅ | ✅ | ✅ | 194.4 mm | 94.3 mm | [PASS](result_images/hand15_result.jpg) |
| [sample11](images/hand16.jpg) | ✅ | ✅ | ✅ | 186.2 mm | 92.0 mm | [PASS](result_images/hand16_result.jpg) |
| [sample12](images/hand17.jpg) | ✅ | ✅ | ✅ | 186.2 mm | 90.5 mm | [PASS](result_images/hand17_result.jpg) |
| [sample13](images/hand18.jpg) | ✅ | ✅ | ✅ | 182.5 mm | 91.2 mm | [PASS](result_images/hand18_result.jpg) |
| [sample14](images/hand19.jpg) | ✅ | ✅ | ✅ | 184.3 mm | 94.6 mm | [PASS](result_images/hand19_result.jpg) |
| [sample15](images/hand20.jpg) | ✅ | ✅ | ✅ | 189.4 mm | 94.6 mm | [PASS](result_images/hand20_result.jpg) |
| [sample16](images/hand21.jpg) | ✅ | ✅ | ✅ | 183.0 mm | 95.4 mm | [PASS](result_images/hand21_result.jpg) |
| [sample17](images/hand24.jpg) | ✅ | ✅ | ✅ | 184.3 mm | 92.8 mm | [PASS](result_images/hand24_result.jpg) |
| [sample18](images/hand25.jpg) | ✅ | ✅ | ✅ | 190.7 mm | 92.2 mm | [PASS](result_images/hand25_result.jpg) |
| [sample19](images/hand14.jpg) | ✅ | ✅ | ❌ | 187.8 mm | FALLBACK | [PARTIAL](result_images/hand14_result.jpg) |


## Measurement accuracy 

The physical reference measurements of the tested hand were:

- Hand length: **185.0 mm**
- Palm width: **94.0 mm**

The fallback sample was excluded from the accuracy calculations.

In the table below, we compare the real world measurement to application's measurement results.

| Metric | Hand length | Palm width |
|---|---:|---:|
| Number of samples | 18 | 18 |
|Real world measurement | 185.0 mm | 94.0 mm |
| Average measured value | 186.7 mm | 93.2 mm |
| Average absolute error | 3.1 mm | 1.3 mm |
| Average percentage error | 1.67% | 1.43% |
| Approx. average accuracy | 98.33% | 98.57% |
| Largest observed error | 9.4 mm | 3.5 mm |

## Inner hand measuring testing

> 10 sample images of inner hand on A4 paper were also tested, and provided successful results. 
> Plain inner hand images are found also at [`images/`](images/)
> Inner hand image results are found at [`result_images/`](result_images/).

[Inner hand result](result_images/handinner3_result.jpg)






