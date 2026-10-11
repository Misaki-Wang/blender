#!/usr/bin/env python3
import argparse,json,math,sys
from pathlib import Path
import bpy
from mathutils import Vector

def args():
 p=argparse.ArgumentParser(); p.add_argument('blend',type=Path); p.add_argument('output',type=Path); p.add_argument('--resolution',type=int,nargs=2,default=[1024,1024]); argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]; return p.parse_args(argv)
def main():
 a=args(); bpy.ops.wm.open_mainfile(filepath=str(a.blend)); out=a.output; out.mkdir(parents=True,exist_ok=True)
 scene=bpy.context.scene; cam=scene.camera
 meshes=[o for o in scene.objects if o.type=='MESH' and o.hide_render is False]
 if not meshes: raise RuntimeError('asset blend needs mesh objects')
 if cam is None:
  seed=[o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
  low=Vector((min(p.x for p in seed),min(p.y for p in seed),min(p.z for p in seed)))
  high=Vector((max(p.x for p in seed),max(p.y for p in seed),max(p.z for p in seed)))
  center=(low+high)*0.5; radius=max((high-low).length,0.1)
  data=bpy.data.cameras.new('AnnotationCamera'); cam=bpy.data.objects.new('AnnotationCamera',data); scene.collection.objects.link(cam); scene.camera=cam
  cam.location=center+Vector((1.6,-1.6,1.1))*radius
  cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler(); data.lens=50

 w,h=a.resolution; scene.render.resolution_x=w; scene.render.resolution_y=h; scene.render.resolution_percentage=100
 try:
  scene.render.engine='BLENDER_EEVEE_NEXT'
 except TypeError:
  scene.render.engine='BLENDER_EEVEE'
 scene.render.image_settings.file_format='PNG'; scene.render.film_transparent=False
 # world and lights remain for RGB; render RGB
 scene.render.filepath=str(out/'rgb.png'); bpy.ops.render.render(write_still=True)
 # bbox in world coordinates
 pts=[o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
 lo=Vector((min(p.x for p in pts),min(p.y for p in pts),min(p.z for p in pts))); hi=Vector((max(p.x for p in pts),max(p.y for p in pts),max(p.z for p in pts)))
 corners=[Vector((x,y,z)) for x in (lo.x,hi.x) for y in (lo.y,hi.y) for z in (lo.z,hi.z)]
 def proj(p):
  q=bpy_extras.object_utils.world_to_camera_view(scene,cam,p); return [q.x*w,(1-q.y)*h,q.z]
 import bpy_extras.object_utils
 uv=[proj(p) for p in corners]; xy=[[p[0],p[1]] for p in uv]; box=[min(x for x,y in xy),min(y for x,y in xy),max(x for x,y in xy),max(y for x,y in xy)]
 # metric depth
 scene.view_layers[0].use_pass_z=True; scene.render.image_settings.file_format='OPEN_EXR'; scene.render.filepath=str(out/'depth.exr'); bpy.ops.render.render(write_still=True)
 # mask by emission override and transparent world
 old=scene.world; scene.world=None
 mats=[]
 for o in meshes:
  mats.append((o,list(o.data.materials))); m=bpy.data.materials.new('asset_mask'); m.diffuse_color=(1,1,1,1); m.use_nodes=True; bs=m.node_tree.nodes.get('Principled BSDF'); bs.inputs['Base Color'].default_value=(1,1,1,1); bs.inputs['Roughness'].default_value=1; o.data.materials.clear(); o.data.materials.append(m)
 scene.render.image_settings.file_format='PNG'; scene.render.film_transparent=True; scene.render.filepath=str(out/'asset_mask.png'); bpy.ops.render.render(write_still=True)
 for o,ms in mats: o.data.materials.clear(); [o.data.materials.append(m) for m in ms]
 scene.world=old
 def mat4(m): return [[float(m[i][j]) for j in range(4)] for i in range(4)]
 unit_scale = float(getattr(scene.unit_settings, 'scale_length', 1.0) or 1.0)
 if not math.isfinite(unit_scale) or unit_scale <= 0:
  unit_scale = 1.0
 world_scale = [hi[i]-lo[i] for i in range(3)]
 world_center = [(lo[i]+hi[i])/2 for i in range(3)]
 metric_scale = [value*unit_scale for value in world_scale]
 metric_center = [value*unit_scale for value in world_center]
 metric_corners = [[float(v)*unit_scale for v in p] for p in corners]
 json.dump({'center':world_center,'scale':world_scale,'center_m':metric_center,'scale_m':metric_scale,'unit_scale_length':unit_scale,'rotation':mat4(cam.matrix_world.to_3x3().to_4x4()),'corners_world':[[float(v) for v in p] for p in corners],'corners_m':metric_corners},open(out/'bbox_3d.json','w'),indent=2)
 json.dump({'xyxy':box,'corners_projected':uv},open(out/'bbox_2d_proj.json','w'),indent=2)
 json.dump({'xyxy':box,'source':'asset_mask.png'},open(out/'bbox_2d_rend.json','w'),indent=2)
 json.dump({'resolution':[w,h],'projection':cam.data.type,'intrinsics':{'focal_length_mm':cam.data.lens,'sensor_width_mm':cam.data.sensor_width},'world_to_camera':mat4(cam.matrix_world.inverted()),'camera_to_world':mat4(cam.matrix_world)},open(out/'camera.json','w'),indent=2)
if __name__=='__main__': main()
