from bsp import BSP


def main():
    cmp = BSP(K=25, h=1, N=4, r=2, p=2)
    try:
        cmp.fit()
        cmp.compute()
        cmp.display_results_3D()
    except Exception as e:
        print(e)
    finally:
        print('Завершение программы')


if __name__ == '__main__':
    main()
