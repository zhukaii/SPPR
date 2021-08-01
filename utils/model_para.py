def filter_para(model, args):
    if args.opt == 'opt1':
        return [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    elif args.opt == 'opt2':
        return [
            {'params': model.backbone.parameters(), 'lr': args.lr},
            {'params': model.embedding.parameters(), 'lr': args.lr},
            {'params': model.new_proto, 'lr': args.lr},
            {"params": model.IL_attn.parameters(), "lr": args.lr},
                ]



