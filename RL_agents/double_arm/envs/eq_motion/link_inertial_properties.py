import argparse

"""
execution: python link_inertial_properties.py -L <value> -H <value> -B <value> -p <value>
e.g. python link_inertial_properties.py -L 1.0 -H 0.05 -B 0.05 -p 2710
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--length", type=float, default=1.0, help="beam length (in meters)")
    parser.add_argument("-H", "--height", type=float, default=0.05, help="beam height (in meters)")
    parser.add_argument("-B", "--breadth", type=float, default=0.05, help="beam breadth (in meters)")
    parser.add_argument("-p", "--density", type=float, default=2710, help="material density (in Kg/m**3)") #aluminum density
    args = parser.parse_args()

    h = args.height
    b = args.breadth
    l = args.length

    p = args.density #density
    v = l*h*b #volume
    m = p*v #mass

    ##about center of mass
    Ixx = m*(h**2 + b**2)/12
    Iyy = m*(l**2 + b**2)/12
    Izz = m*(l**2 + h**2)/12
    print('mass: ', m)
    print('Ixx: ', Ixx)
    print('Iyy: ', Iyy)
    print('Izz: ', Izz)

if __name__ == "__main__":
    main()
