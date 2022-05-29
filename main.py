from bsp import BSP


def main():
    time = []
    for i in range(8):
        cmp = BSP(K=5, h=1, N=4, r=0, p=i)
        try:
            cmp.fit()
            # cmp.display_psi()
            cmp.naive_compute_by_map()
        except Exception as e:
            print(e)
        finally:
            time.append(cmp.get_time())
            # cmp.computation_time()
            # cmp.display_results_2D()
    print(time)

if __name__ == '__main__':
    main()
