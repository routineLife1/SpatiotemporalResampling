# Absolutely Eliminate Anime Stutters By SpatiotemporalResampling
![2](https://github.com/hyw-dev/SpatiotemporalResampling/assets/68835291/ea34db54-d717-499e-9861-55e02a7413af)


## Using the currently best performing open-source animation vfi algorithm ([GMFSS_UNION](https://github.com/98mxr/GMFSS_union)) and the idea of SpatiotemporalResampling, completely eliminate anime vfi stutters caused by repeated animation character frames (one shot two and one shot three)

## Usage
1. Set the video path at the "video="
2. Set the save path at the "save="
3. Set the output frame resolution at the "global_size="
4. Set the resampe times at the "times=" (output_fps = input_fps / 2 * times)
5. Choose whether to enable scene change detection at the "scene_det=" (True/False) (If you want to ensure smoothness, please set scene_det to False)

## Contribution
1. Solved the problem of reduced accuracy caused by occlusion of characters in traditional one shot N recognition algorithms
2. Solved the problem of inaccurate recognition of rhythm when the proportion of characters in the screen is small and the overall movement of the screen is large
3. Solved the problem of multiple characters with different rhythm in the same screen
4. The method is also applicable to one shot two/three, two/three shot interlaced (simple and efficient)
5. Avoiding significant degradation in exported frame quality caused by misjudgment in one shot two/one shot three recognition
6. The method is applicable to any animation vfi algorithm

##  Limitation
1. The processing results strongly depend on the strength of the vfi algorithm
2. The input video should meet the frame rate of commonly used anime videos (23.976)
3. The method cannot handle one shot four/five or ... which not commonly seen in anime production
4. In an ideal state, the overall frame sequence is offset back by 0.5 frame times. In the case of imperfect vfi algorithms, the offset is uncertain (but not more than 1.0 frame times)
5. Reduced the quality of the original one shot one segment

