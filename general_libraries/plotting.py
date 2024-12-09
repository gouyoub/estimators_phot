import matplotlib.pyplot as plt

def plot_auto_Cl(cl1, cl2, ell, err, nzbins, size, probe, nx, ny, label1='', label2=''):

    k1 = {'GCph':'D', 'WL':'G', 'GGL':'D'}
    k2 = {'GCph':'D', 'WL':'G', 'GGL':'G'}

    fig = plt.figure(figsize=size)
    for (idx,i) in enumerate(range(1,nzbins+1)):
        k = f'{k1[probe]}{idx}-{k2[probe]}{idx}'
        ax = fig.add_subplot(nx,ny,i)
        ax.set_title('redshift bin {}'.format(i))

        ax.loglog()
        ax.errorbar(ell, cl1[k], yerr=err[k], ls='none', elinewidth=1.5, marker='s', c='black', label=label1, ms=2)
        ax.plot(ell, cl2[k], label=label2)

        ax.set_xlabel('$\ell$', fontsize=15)
        ax.set_ylabel('$C_\ell$')

        # ax.set_xlim([30,2500])
        if i==1:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.5), ncol=3)

    # fig.tight_layout()
    # plt.savefig('plots/theo-vs-estim/lbias_nlbias_firstchains.png')
    plt.show()

def plot_cross_sym_Cl(cl1, cl2, ell, err, nzbins, size, probe, label1='', label2=''):

    k1 = {'GCph':'D', 'WL':'G', 'GGL':'D'}
    k2 = {'GCph':'D', 'WL':'G', 'GGL':'G'}

    fig = plt.figure(figsize=size)
    c = 0
    for i in range(nzbins):
        for j in range(i,nzbins):
            ax = fig.add_subplot(nzbins, nzbins, i+nzbins*j+1)
            k = f'{k1[probe]}{i}-{k2[probe]}{j}'

            ax.loglog()
            ax.errorbar(ell, cl1[k], yerr=err[k], ls='none', elinewidth=1.5, marker='s', c='black', label=label1, ms=2)
            ax.plot(ell, cl2[k], label=label2)

            if i == 0:
                ax.set_ylabel('$C_\ell$', fontsize=20)
            if j == nzbins-1:
                ax.set_xlabel('$\ell$', fontsize=20)
            ax.axhline(1, ls='--', c='black')
            if c==0:
                ax.legend(loc='upper left', bbox_to_anchor=(0, 1.4))
            c+=1

    fig.tight_layout()
    # plt.savefig('plots/bf/WL_fit_bf.png')
    plt.show()

def plot_cross_nonsym_Cl(cl1, cl2, ell, err, nzbins, size, probe, label1='', label2=''):

    k1 = {'GCph':'D', 'WL':'G', 'GGL':'D'}
    k2 = {'GCph':'D', 'WL':'G', 'GGL':'G'}

    fig = plt.figure(figsize=size)
    c = 0
    for i in range(nzbins):
        for j in range(nzbins):
            ax = fig.add_subplot(nzbins, nzbins, i+nzbins*j+1)
            k = f'{k1[probe]}{i}-{k2[probe]}{j}'

            ax.loglog()
            ax.errorbar(ell, cl1[k], yerr=err[k], ls='none', elinewidth=1.5, marker='s', c='black', label=label1, ms=2)
            ax.plot(ell, cl2[k], label=label2)

            if i == 0:
                ax.set_ylabel('$C_\ell$', fontsize=20)
            if j == nzbins-1:
                ax.set_xlabel('$\ell$', fontsize=20)
            ax.axhline(1, ls='--', c='black')
            if c==0:
                ax.legend(loc='upper left', bbox_to_anchor=(0, 1.4))
            c+=1

    fig.tight_layout()
    # plt.savefig('plots/bf/WL_fit_bf.png')
    plt.show()