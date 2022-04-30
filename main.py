from bsp import BSP


def main():
    cmp = BSP(K=25, h=1, N=4, r=0, p=2)
    try:
        cmp.fit()
        cmp.compute_by_map()
    except Exception as e:
        print(e)
    finally:
        cmp.computation_time()
        cmp.display_results_2D()


if __name__ == '__main__':
    main()
