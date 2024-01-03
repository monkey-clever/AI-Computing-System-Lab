# 智能计算系统-实验-答案-7-1-YOLOv3

#### 介绍
智能计算系统-实验-答案-7-1-YOLOv3

80分

> 60 分标准：补全 nms_detection.h 文件，cnplugin 可正常编过。\
> 70 分标准：在 60 分的基础上完成 TensorFlow 的集成编译，完成 pb 模型添加后处理大算子的操作，执行测试脚本时 map 值高于 30%，单 batch 延时（包含后处理）低于300ms。\
> 80 分标准：在 70 分的基础上，执行测试脚本时 map 值高于 50%，单 batch 延时（包含后处理）低于 100ms。\
> 90 分标准：在 80 分的基础上，执行测试脚本时 map 值高于 54%，单 batch 延时（包含后处理）低于 50ms。\
> 100 分标准：在 90 分的基础上，执行测试脚本时 map 值高于 56%，单 batch 延时（包含后处理）低于 25ms。\


#### Tree


```
智能计算系统-实验-答案-7-1-YOLOv3
└── yolov3
    ├── bangc
    │   └── PluginYolov3DetectionOutputOp
    │       ├── BANG_LOG.h
    │       ├── cnplugin.h
    │       ├── nms_detection.h
    │       ├── plugin_yolov3_detection_helper.h
    │       ├── plugin_yolov3_detection_output_kernel_v1
    │       ├── plugin_yolov3_detection_output_kernel_v1.h
    │       ├── plugin_yolov3_detection_output_kernel_v2.mlu
    │       └── plugin_yolov3_detection_output_op.cc
    ├── readme.txt
    ├── tf-implementation
    │   └── tf-1.14-detectionoutput
    │       ├── BUILD
    │       ├── image_ops.cc
    │       ├── mlu_lib_ops.cc
    │       ├── mlu_lib_ops.h
    │       ├── mlu_ops.h
    │       ├── mlu_stream.h
    │       ├── readme.txt
    │       ├── yolov3_detection_output_op.cc
    │       ├── yolov3_detection_output_op_mlu.h
    │       └── yolov3detectionoutput.cc
    └── yolov3-bcl
        └── demo
            └── yolov3_int8.pbtxt
```


#### 参考

1.  开发手册下载：[文档中心 – 寒武纪开发者社区](http://developer.cambricon.com/index/document/index/classid/3.html)
2.  实验教程：[《智能计算系统》实验-7-1-YOLOv3](http://blog.csdn.net/weixin_40943865/article/details/122059436)
3.  [TensorFlow的自定义算子实现](http://blog.csdn.net/weixin_40943865/article/details/122225775)
4.  [BANGC_Yolov3_Tutorial](https://github.com/CambriconECO/BANGC_Yolov3_Tutorial)

