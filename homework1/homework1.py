__author__ = 'rim'

import argparse
import math
import pylab


COLORS = ['r', 'g', 'b', 'm', 'c', 'y', 'k', '#eaefff']


def distance_measure(x, y):
    if len(x) != len(y):
        raise
    s = 0
    for i in xrange(0, len(x)):
        s += (x[i] - y[i])**2
    return math.sqrt(s)


def dbscan(xs, labels, eps, min_pts):
    nv = xs[:]
    for nvi in nv:
        i = xs.index(nvi)
        list.remove(nv, nvi)
        nbr = [x for x in xs if distance_measure(nvi, x) < eps]
        if len(nbr) < min_pts:
            labels[i] = False
        else:
            c = []
            labels[i] = True
            expand_cluster(nvi, xs, nbr, c, eps, min_pts, nv, labels)
            yield c

def expand_cluster(x, xs, nbr, c, eps, min_pts, nv, labels):
    c.append(x)
    for x1 in nbr:
        if x1 in nv:
            list.remove(nv, x1)
            nbr1 = [x2 for x2 in xs if distance_measure(x1, x2) < eps]
            if len(nbr1) >= min_pts:
                nbr = nbr + nbr1
                result = []
                map(lambda elem: not elem in result and result.append(elem), nbr)
                nbr = result
        i = xs.index(x1)
        if labels[i] != True:
            c.append(x1)
            labels[i] = True

def rand_index(ideal_clusterization, my_clusterization):
    n = len(my_clusterization)
    a, b = 0, 0
    n1 = len(ideal_clusterization)
    c = n1*(n1-1)/2.0
    for i in xrange(0, n):
        for j in xrange(i+1, n):
            if(ideal_clusterization[i]==ideal_clusterization[j] and my_clusterization[i]==my_clusterization[j]): a += 1
            if(ideal_clusterization[i]!=ideal_clusterization[j] and my_clusterization[i]!=my_clusterization[j]): b += 1
    return (a+b)/c

def average(cs):
    s = 0.0
    count = 0.0
    for c in cs:
        for i in xrange(0, len(c)):
            for j in xrange(i, len(c)):
                s += distance_measure(c[i], c[j])
                count += 1
    if count > 0:
        return s/count
    else:
        return 0.0


def print_info(xs, cs, labels, ideal_clusterization, my_clusterization):
    print "number of x", len(labels)
    print "number of attr", len(xs[0])
    for c in enumerate(cs):
        print c

    print "ideal_clusterization"
    for x in ideal_clusterization:
        print x,
    print
    #for x in xs:
     #   print x


    print "\nmy_clusterization"
    for x in my_clusterization:
        print x,
    print

    for l in labels:
        if l != True:
            print l,

    print "\nRand Index",rand_index(ideal_clusterization, my_clusterization)




def main():
    args = parse_args()
    # input matrix X == xs
    xs = []
    with open(args.data_path[0],"r") as data_file:
        for line in data_file:
            list.insert(xs, 0, line.split(","))
    for i in xrange(0, len(xs)):
        xs[i] = [float(x) for x in xs[i]]
    ideal_clusterization = [int(line[0]) for line in xs]
    labels = [None for line in xs] #False - noise, True - in cluster, None - not in cluster, not visited
    xs = [row[1:] for row in xs]
    maxs = [p for p in xs[0]]

    #normalization
    for i in xrange(0,len(xs)):
        for j in xrange(0, len(maxs)):
            if maxs[j] < xs[i][j]:
                maxs[j] = xs[i][j]
    for i in xrange(0, len(xs)):
        for j in xrange(0, len(maxs)):
            xs[i][j] /= maxs[j]



    cs = dbscan(xs, labels, args.eps, args.min_pts)
    cs = list(cs)
    my_clusterization = []
    for x in xs:
        for i, c in enumerate(cs):
            if x in c:
                my_clusterization.append(i)

    print_info(xs,cs,labels, ideal_clusterization, my_clusterization)

    if bool(args.plot_eps):
        def drange(start, stop, step):
            r = start
            while r < stop:
                yield r
                r += step
        eps = list(drange(0.2, 1, 0.01))
        rands = []
        ls = []
        clusters = []
        avrgs = []
        for e in eps:
            labels = [False for l in labels]
            cs = dbscan(xs, labels, e, args.min_pts)
            cs = list(cs)
            my_clusterization = []
            for x in xs:
                for i, c in enumerate(cs):
                    if x in c:
                        my_clusterization.append(i)
            rands.append(rand_index(ideal_clusterization, my_clusterization))
            ls.append(len([x for x in labels if x != True]))
            clusters.append(len(cs))
            avrgs.append(average(cs))

        pylab.figure(figsize=(30, 10))
        pylab.subplot(1,4,1)
        pylab.plot(eps, clusters, 'b.', ls='-')
        pylab.title('curve for eps-numberOfClusters')
        pylab.ylabel("number of clusters")
        pylab.plot(eps, [3 for e in eps], 'r', ls='--')

        pylab.subplot(1,4,2)
        pylab.plot(eps, rands, 'b*', ls='-')
        pylab.title('curve for eps-randIndex')
        pylab.xlabel("value of eps")
        pylab.ylabel("rand index")

        pylab.subplot(1,4,3)
        pylab.plot(eps, avrgs, 'b*', ls='-')
        pylab.title('curve for eps-averageDistance')
        pylab.xlabel("value of eps")
        pylab.ylabel("averageDistance")


        pylab.subplot(1,4,4)
        pylab.plot(eps, ls, 'r.', ls='-')
        pylab.title('curve for eps-noises \nnumber of all elements =%d' % len(xs))
        pylab.ylabel('number of noises')
        pylab.show()



def parse_args():
    parser = argparse.ArgumentParser(description='DBSCAN')
    parser.add_argument('-e', dest='eps', help='epsilon', type=float, default=0.85)
    parser.add_argument('-m', dest='min_pts', help='minimum number of points required to form a dense region', type=float, default=15)
    parser.add_argument('-p', dest='plot_eps', help='plot curve for optimal eps', type=bool, default=False)
    parser.add_argument('data_path', nargs=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()