# lst-super-res
Increase the resolution of Land Surface Temperate (LST) images.

My theoretical file structure: 

code -- run in AWS EC2
- data_processing
  - tile
  - aggregate

data -- in AWS S3
- raw
  - LST
  - RGB
- tiles
  - LST
  - RGB
- aggregated
  - input_LST
  - output_LST
