import pickle
import numpy as np
import logging

def load_variavle(filename):
   f=open(filename,'rb')
   r=pickle.load(f)
   f.close()
   return r
# img_backbone
print('Forward img_backbone compare')
for i in range(4):
   paddle_var = load_variavle('paddle_var/img_backbone_feats_{}.txt'.format(i))
   torch_var = load_variavle('torch_var/img_backbone_feats_{}.txt'.format(i))
   compare = np.abs(paddle_var - torch_var).max()
   print('backbone feat_{}, compare:{}'.format(i, compare))

print('Forward img_neck compare')
# img_neck
for i in range(4):
   paddle_var = load_variavle('paddle_var/img_neck_feats_{}.txt'.format(i))
   torch_var = load_variavle('torch_var/img_neck_feats_{}.txt'.format(i))
   compare = np.abs(paddle_var - torch_var).max()
   print('neck feat_{}, compare:{}'.format(i, compare))


paddle_var = load_variavle('paddle_var/points.txt')
torch_var = load_variavle('torch_var/points.txt')
compare = np.abs(paddle_var - torch_var).max()
print('points compare:{}'.format(compare))

paddle_var = load_variavle('paddle_var/voxels.txt')
torch_var = load_variavle('torch_var/voxels.txt')
compare = np.abs(paddle_var - torch_var).max()
print('voxels compare:{}'.format(compare))

paddle_var = load_variavle('paddle_var/pts_middle_encoder.txt')
torch_var = load_variavle('torch_var/pts_middle_encoder.txt')
compare = np.abs(paddle_var - torch_var).max()
print('pts_middle_encoder compare:{}'.format(compare))

for i in range(2):
   paddle_var = load_variavle('paddle_var/pts_backbone_{}.txt'.format(i))
   torch_var = load_variavle('torch_var/pts_backbone_{}.txt'.format(i))
   compare = np.abs(paddle_var - torch_var).max()
   print('pts_backbone feat_{}, compare:{}'.format(i, compare))

for i in range(4):
   paddle_var = load_variavle('paddle_var/pts_neck_{}.txt'.format(i))
   torch_var = load_variavle('torch_var/pts_neck_{}.txt'.format(i))
   compare = np.abs(paddle_var - torch_var).max()
   print('pts_neck feat_{}, compare:{}'.format(i, compare))

# head_out
print('Forward head compare')
paddle_var = load_variavle('paddle_var/out_all_bbox_preds.txt')
torch_var = load_variavle('torch_var/out_all_bbox_preds.txt')
compare = np.abs(paddle_var - torch_var).max()
print('out_all_bbox_preds compare:{}'.format(compare))
paddle_var = load_variavle('paddle_var/out_all_cls_scores.txt')
torch_var = load_variavle('torch_var/out_all_cls_scores.txt')
compare = np.abs(paddle_var - torch_var).max()
print('out_all_cls_scores compare:{}'.format(compare))

# back img_backbone
print('Backward img_backbone compare')
for i in range(4):
   paddle_var = load_variavle('paddle_var/b_img_backbone_feats_{}.txt'.format(i))
   torch_var = load_variavle('torch_var/b_img_backbone_feats_{}.txt'.format(i))
   compare = np.abs(paddle_var - torch_var).max()
   print(compare)

# back img_neck
print('Backward img_neck compare')
for i in range(4):
   paddle_var = load_variavle('paddle_var/b_img_neck_feats_{}.txt'.format(i))
   torch_var = load_variavle('torch_var/b_img_neck_feats_{}.txt'.format(i))
   compare = np.abs(paddle_var - torch_var).max()
   print(compare)

paddle_var = load_variavle('paddle_var/points.txt')
torch_var = load_variavle('torch_var/points.txt')
compare = np.abs(paddle_var - torch_var).max()
print('points compare:{}'.format(compare))

paddle_var = load_variavle('paddle_var/voxels.txt')
torch_var = load_variavle('torch_var/voxels.txt')
compare = np.abs(paddle_var - torch_var).max()
print('voxels compare:{}'.format(compare))

paddle_var = load_variavle('paddle_var/pts_middle_encoder.txt')
torch_var = load_variavle('torch_var/pts_middle_encoder.txt')
compare = np.abs(paddle_var - torch_var).max()
print('pts_middle_encoder compare:{}'.format(compare))

for i in range(2):
   paddle_var = load_variavle('paddle_var/b_pts_backbone_{}.txt'.format(i))
   torch_var = load_variavle('torch_var/b_pts_backbone_{}.txt'.format(i))
   compare = np.abs(paddle_var - torch_var).max()
   print('pts_backbone feat_{}, compare:{}'.format(i, compare))

for i in range(4):
   paddle_var = load_variavle('paddle_var/b_pts_neck_{}.txt'.format(i))
   torch_var = load_variavle('torch_var/b_pts_neck_{}.txt'.format(i))
   compare = np.abs(paddle_var - torch_var).max()
   print('pts_neck feat_{}, compare:{}'.format(i, compare))

# back head_out
print('Backward head out compare')
paddle_var = load_variavle('paddle_var/b_out_all_bbox_preds.txt')
torch_var = load_variavle('torch_var/b_out_all_bbox_preds.txt')
compare = np.abs(paddle_var - torch_var).max()
print(compare)
paddle_var = load_variavle('paddle_var/b_out_all_cls_scores.txt')
torch_var = load_variavle('torch_var/b_out_all_cls_scores.txt')
compare = np.abs(paddle_var - torch_var).max()
print(compare)


