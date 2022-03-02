from micronet_ressources.profile import profile

def score(model):
    ref_params = 5586981
    ref_flops  = 834362880

    flops, params = profile(model, (1,3,32,32))
    flops, params = flops.item(), params.item()

    score_flops = flops / ref_flops
    score_params = params / ref_params
    score = score_flops + score_params
    print("Flops: {}, Params: {}".format(flops,params))
    print("Score flops: {} Score Params: {}".format(score_flops,score_params))
    print("Final score: {}".format(score))
    return score