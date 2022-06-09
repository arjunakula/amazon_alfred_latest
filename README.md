ALFRED-L dataset and code release

Rendering new trajectories:

python -m alfred.gen.render_trajs


Create LMDB files:

python -m alfred.data.create_lmdb with args.visual_checkpoint=$ET_LOGS/pretrained/fasterrcnn_model.pth args.data_output=lmdb_human_unseen args.vocab_path=$ET_ROOT/files/human.vocab


Evaluation:

python -m alfred.eval.eval_agent with eval.exp=pretrained eval.checkpoint=et_human_pretrained.pth eval.object_predictor=$ET_LOGS/pretrained/maskrcnn_model.pth exp.num_workers=5 eval.eval_range=None exp.data.valid=lmdb_human eval.split=valid_seen
