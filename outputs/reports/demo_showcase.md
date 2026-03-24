# Demo Showcase

## DeepSolarEye Cases

### deepsolareye_1
- sample_id: deepsolareye:solar_Wed_Jun_21_17__49__0_2017_L_0.0172684458399_I_0.0549568627451
- image_path: C:\Users\wailo\Desktop\solar\data\raw\deepsolareye\SolarPanelSoilingImageDataset\Solar_Panel_Soiling_Image_dataset\PanelImages\solar_Wed_Jun_21_17__49__0_2017_L_0.0172684458399_I_0.0549568627451.jpg
- summary: Predicted power loss 3.39%
- final_severity: low
- recommended_action: Monitor panel condition
- priority: monitor

### deepsolareye_2
- sample_id: deepsolareye:solar_Wed_Jun_14_15__12__46_2017_L_0.115371507661_I_0.563839215686
- image_path: C:\Users\wailo\Desktop\solar\data\raw\deepsolareye\SolarPanelSoilingImageDataset\Solar_Panel_Soiling_Image_dataset\PanelImages\solar_Wed_Jun_14_15__12__46_2017_L_0.115371507661_I_0.563839215686.jpg
- summary: Predicted power loss 11.71%
- final_severity: low
- recommended_action: Monitor panel condition
- priority: monitor

### deepsolareye_3
- sample_id: deepsolareye:solar_Wed_Jun_21_14__19__50_2017_L_0.374342112975_I_0.669105882353
- image_path: C:\Users\wailo\Desktop\solar\data\raw\deepsolareye\SolarPanelSoilingImageDataset\Solar_Panel_Soiling_Image_dataset\PanelImages\solar_Wed_Jun_21_14__19__50_2017_L_0.374342112975_I_0.669105882353.jpg
- summary: Predicted power loss 19.08%
- final_severity: high
- recommended_action: Schedule cleaning and performance inspection
- priority: schedule_check

### deepsolareye_4
- sample_id: deepsolareye:solar_Wed_Jun_21_8__34__18_2017_L_0.327396573585_I_0.398976470588
- image_path: C:\Users\wailo\Desktop\solar\data\raw\deepsolareye\SolarPanelSoilingImageDataset\Solar_Panel_Soiling_Image_dataset\PanelImages\solar_Wed_Jun_21_8__34__18_2017_L_0.327396573585_I_0.398976470588.jpg
- summary: Predicted power loss 35.44%
- final_severity: urgent
- recommended_action: Schedule cleaning and performance inspection
- priority: schedule_check

### deepsolareye_5
- sample_id: deepsolareye:solar_Fri_Jun_30_9__33__57_2017_L_0.924377719316_I_0.310054901961
- image_path: C:\Users\wailo\Desktop\solar\data\raw\deepsolareye\SolarPanelSoilingImageDataset\Solar_Panel_Soiling_Image_dataset\PanelImages\solar_Fri_Jun_30_9__33__57_2017_L_0.924377719316_I_0.310054901961.jpg
- summary: Predicted power loss 83.57%
- final_severity: urgent
- recommended_action: Schedule cleaning and performance inspection
- priority: schedule_check


## Villegas Cases

### villegas_1
- sample_id: 2020_16_04_10_49
- image_path: C:\Users\wailo\Desktop\solar\data\interim\villegas\images\2020_16_04_10_49.jpg
- summary: Predicted pmpp 39.5568, isc 5.4376, ff 0.7829
- final_severity: low
- recommended_action: Monitor panel condition
- priority: monitor

### villegas_2
- sample_id: 2020_07_04_08_57
- image_path: C:\Users\wailo\Desktop\solar\data\interim\villegas\images\2020_07_04_08_57.jpg
- summary: Predicted pmpp 9.7667, isc 0.7735, ff 0.7169
- final_severity: medium
- recommended_action: Check shading conditions and electrical behavior
- priority: monitor

### villegas_3
- sample_id: 2021_13_10_17_05
- image_path: C:\Users\wailo\Desktop\solar\data\interim\villegas\images\2021_13_10_17_05.jpg
- summary: Predicted pmpp 7.5565, isc -0.0459, ff 0.4227
- final_severity: medium
- recommended_action: Check shading conditions and electrical behavior
- priority: monitor


## TRSAI Cases

### trsai_1
- sample_id: v2:img20220823_142035_bmp.rf.f5056a1fe9a07916c652134550532546
- image_path: C:\Users\wailo\Desktop\solar\data\interim\trsai\v2\test\images\img20220823_142035_bmp.rf.f5056a1fe9a07916c652134550532546.jpg
- summary: Predicted hotspot probability 0.9957
- final_severity: urgent
- recommended_action: Immediate inspection for hotspot risk
- priority: high_priority

### trsai_2
- sample_id: v2:img20220907_102928_bmp.rf.c1fed4204268e7d69adb5f37525114aa
- image_path: C:\Users\wailo\Desktop\solar\data\interim\trsai\v2\test\images\img20220907_102928_bmp.rf.c1fed4204268e7d69adb5f37525114aa.jpg
- summary: Predicted hotspot probability 0.9992
- final_severity: urgent
- recommended_action: Immediate inspection for hotspot risk
- priority: high_priority

### trsai_3
- sample_id: v1:img20220823_130946-2-_bmp.rf.919992c77676b3351a2ef05fb30dd674
- image_path: C:\Users\wailo\Desktop\solar\data\interim\trsai\v1\test\images\img20220823_130946-2-_bmp.rf.919992c77676b3351a2ef05fb30dd674.jpg
- summary: Predicted hotspot probability 0.9998
- final_severity: urgent
- recommended_action: Immediate inspection for hotspot risk
- priority: high_priority

## Manual Fusion Showcase

### fusion_monitor_case
- power_loss_pct: 3.385273694992065
- electrical_score: N/A
- hotspot_probability: N/A
- final_severity: low
- final_score: 0.0609
- recommended_action: Monitor panel condition
- priority: monitor
- source_notes: Low-power-loss DeepSolarEye sample deepsolareye:solar_Wed_Jun_21_17__49__0_2017_L_0.0172684458399_I_0.0549568627451

### fusion_cleaning_case
- power_loss_pct: 83.57443237304688
- electrical_score: N/A
- hotspot_probability: N/A
- final_severity: urgent
- final_score: 0.4500
- recommended_action: Schedule cleaning and performance inspection
- priority: schedule_check
- source_notes: High-power-loss DeepSolarEye sample deepsolareye:solar_Fri_Jun_30_9__33__57_2017_L_0.924377719316_I_0.310054901961

### fusion_electrical_case
- power_loss_pct: N/A
- electrical_score: 0.5945989129513297
- hotspot_probability: N/A
- final_severity: medium
- final_score: 0.1784
- recommended_action: Check shading conditions and electrical behavior
- priority: monitor
- source_notes: Representative Villegas sample 2020_07_04_08_57

### fusion_hotspot_case
- power_loss_pct: N/A
- electrical_score: N/A
- hotspot_probability: 0.9998290538787842
- final_severity: urgent
- final_score: 0.5999
- recommended_action: Immediate inspection for hotspot risk
- priority: high_priority
- source_notes: High-probability TRSAI sample v1:img20220823_130946-2-_bmp.rf.919992c77676b3351a2ef05fb30dd674

### fusion_combined_urgent_case
- power_loss_pct: 83.57443237304688
- electrical_score: 0.5945989129513297
- hotspot_probability: 0.9998290538787842
- final_severity: urgent
- final_score: 1.0000
- recommended_action: Immediate inspection for hotspot risk
- priority: urgent
- source_notes: Combined showcase from DeepSolarEye deepsolareye:solar_Fri_Jun_30_9__33__57_2017_L_0.924377719316_I_0.310054901961, Villegas 2020_07_04_08_57, TRSAI v1:img20220823_130946-2-_bmp.rf.919992c77676b3351a2ef05fb30dd674
