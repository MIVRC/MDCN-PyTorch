def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.lr_decay = 100

    if args.template.find('EDSR') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDCN') >= 0:
        args.model = 'MDCN'
        args.n_blocks = 12
        args.n_feats = 128
        args.patch_size = 48
        args.chop = True
