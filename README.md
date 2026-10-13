# RTAdapter: Exploring the Upper Bound of In-Context Segmentation from a Data-Centric Perspective
The Code for In-Context Segmentation Paper RTAdapter

sqadapter = SQAdapter(self.ref_masks_pool, self.nshot, self.device, self.args)
ref_feats_prime = sqadapter.transform(ref_feats, tar_feat)

add new args:
config_path = f"adapter_config/{args.benchmark}.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"配置文件不存在：{config_path}，请检查路径和benchmark参数！")
with open(config_path, 'r', encoding='utf-8') as f:
    adapter_config = yaml.safe_load(f)
for key, value in adapter_config.items():
    setattr(args, key, value)
    
