def run_func(cfg):
    from nas4candle.nasapi.benchmark.benchmark_functions_wrappers import polynome_2
    f, (a, b), optimum = polynome_2()
    x = cfg['x']
    res = f(x)
    print(f'res: {res}')
    return res
