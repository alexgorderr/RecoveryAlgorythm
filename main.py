from bsp import BSP


def main():
    cmp = BSP(K=25, h=1, N=20, r=2, p=3)
    try:
        cmp.fit()
        cmp.improved_compute_by_map()
    except Exception as e:
        print(e)
    finally:
        cmp.computation_time()
        cmp.display_results_2D()


if __name__ == '__main__':
    main()
