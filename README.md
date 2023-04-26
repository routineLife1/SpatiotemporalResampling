# SpatiotemporalResampling
Using an effective spatiotemporal resampling VFI method (based on [GMFSS_UNION](https://github.com/98mxr/GMFSS_union)) to remove duplicate animation character frames and improve video frame rate

使用条件:
输入视频素材帧率在23.976(24000/1001)左右，基本只含有一拍二，一拍三画面（不含有一拍四等更高节拍的画面）

优点:
除重本身不受遮挡问题的影响, 除重绝对彻底, 方法简单高效

缺点:
十分考验补帧网络性能, 在补帧绝对无错误的情况下仍然会丢失部分信息, 画面整体节律向前偏移

贡献:
提供了一种简单高效的动漫一拍二，一拍三补帧处理方法
