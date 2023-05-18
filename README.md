## Absolutely Eliminate Anime Stutters By SpatiotemporalResampling
![1](https://github.com/hyw-dev/SpatiotemporalResampling/assets/68835291/efa075c2-56e6-4827-a2cc-7fe4ff16c291)


### Using the currently best performing open-source animation vfi algorithm ([GMFSS_UNION](https://github.com/98mxr/GMFSS_union)) and the idea of SpatiotemporalResampling, completely eliminate anime vfi stutters caused by repeated animation character frames (one shot two and one shot three)

### Usage
1. Install the requirement of [GMFSS](https://github.com/hyw-dev/GMFSS)
2. Download the [weights](https://drive.google.com/drive/folders/1ghfxbyB4vWmcm4qKKFIzAEOI9CXiI9wq?usp=share_link) and put it to the train_logs folder
3. Set the video path at the "video="
4. Set the save path at the "save="
5. Set the output frame resolution at the "global_size="
6. Set the resampe times at the "times=" (output_fps = input_fps / 2 * times)
7. Choose whether to enable scene change detection at the "scene_det=" (True/False) (If you want to ensure smoothness, please set scene_det to False)
8. Run the following command
   > python3 resample.py
10. Turn output frame sequenece into video using ffmpeg (set fps to output_fps)

### Features
1. Avoiding reduced accuracy caused by occlusion of characters in traditional one shot N recognition algorithms
2. Avoiding inaccurate recognition of rhythm when the proportion of characters in the screen is small and the overall movement of the screen is large
3. Solved the problem of multiple characters with different rhythm in the same screen
4. The method is also applicable to one shot two/three, two/three shot interlaced (simple and efficient)
5. Avoiding significant degradation in exported frame quality caused by misjudgment in one shot two/one shot three recognition
6. The method is applicable to any animation vfi algorithm

###  Limitation
1. The processing results strongly depend on the strength of the vfi algorithm
2. The input video should meet the frame rate of commonly used anime videos (23.976)
3. The method cannot handle one shot four/five or ... which not commonly seen in anime production
4. In an ideal state, the overall frame sequence is offset back by 0.5 frame times. In the case of imperfect vfi algorithms, the offset is uncertain (but not more than 1.0 frame times)
5. Reduced the quality of the original one shot one segment

