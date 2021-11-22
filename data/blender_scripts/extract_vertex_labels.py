
import bpy

ob = bpy.context.object
obdata = bpy.context.object.data

label = []
for v in obdata.vertices:
    if bpy.context.object.vertex_groups['roof'].index in [i.group for i in v.groups]:
        label.append(str(1))
    else:
        label.append(str(0))

with open('/home/ihahanov/Projects/roof-measurements/dl_roof_extraction/meshcnn'
          '/datasets/roof_seg/vseg/2510 Garrison Simplified + Annotated.eseg', 'w') as f:
    f.write('\n'.join(label))
