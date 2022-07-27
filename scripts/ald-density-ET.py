"""
Plot Experimental vs Theoretical Densities

https://docs.google.com/spreadsheets/d/1CnYIYPMymwAKaVlElBk4ceNN2RtTxpKIHTzuTRjfY3s/edit#gid=1436201923
"""


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import random
from adjustText import adjust_text
import re

sheet_name_exp = "FilmProps"
sheet_name_thr = "TheorDensity"
sheet_id = "1CnYIYPMymwAKaVlElBk4ceNN2RtTxpKIHTzuTRjfY3s"
url_exp = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_exp}"
url_thr = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_thr}"

# url_exp = "data/data_exp.csv"
# url_thr = "data/data_thr.csv"

def ald_print_info(df, label="", info=True, head=[True, 5], tail=[False, 5], cols=True, vc=[False, None]):
    # VIEW DATA
    print()
    print(">"*60, f" {label}") # visual separator on terminal
    if cols:
        print(list(df.columns), "\n") # view column names

    if info:
        print(df.info(), "\n") # view column names and types

    if head[0]:
        print(df.head(head[1]), "\n") # view the first few rows of the dataframe

    if tail[0]:
        print(df.tail(tail[1]), "\n") # view the last few rows of the dataframe

    if vc[0]:
        print(df[vc[1]].value_counts())

    print("<"*60) # visual separator on terminal


def ald_load_data(urls, view_data=True):

    df = 0
    for url in urls:
        df = pd.read_csv(url)

        # CLEAN DATA
        df.dropna(axis='columns', how='all', inplace=True) # drop columns that are all NA values
        df.columns = df.columns.str.replace(r'[ ]{2,}',' ', regex=True) # make sure column names only have 1 space between words

        if view_data:

            # VIEW DATA
            sheet = url.split("sheet=", 1)[1]
            ald_print_info(df, label=sheet)

        yield df


data_generator = ald_load_data([url_exp, url_thr], view_data=False)
df_exp = next(data_generator)
df_thr = next(data_generator)

# GET DATA TO PLOT

def ald_merge_df(df1, df2, key, cols, view_data=True):
    exp_density = df1.loc[:, [key, cols[0][0]]]
    thr_density = df2.loc[:, [key, cols[1][0]]]
    thr_phase = df2.loc[:, [key, *cols[1]]]

    df_merged = pd.merge(exp_density, thr_density, on=key)
    df_merged = pd.merge(df_merged, thr_phase, on=[key, cols[1][0]])

    if view_data:
        ald_print_info(df_merged, label=" MERGED")

    return df_merged


# PRINT COLUMN NAMES
exp_all_columns = list(df_exp.columns)
print("EXPERIMENTAL ",exp_all_columns)


thr_all_columns = list(df_thr.columns)
print("THEORETICAL ",thr_all_columns)


# RENAME DENSITY COLUMNS
density_exp = "Experimental Density"
density_thr = "Theoretical Density"
density_units = "$(g.cm^{-3})$"

re_cols = {
    "exp": {
        "Density (g.cm-3)": density_exp,
        "Tdep (°C)": "Deposition Temperature (°C)"
    }, 
    "thr": {"Density gcm3": density_thr}
}

df_exp.rename(columns=re_cols['exp'], inplace=True)
df_thr.rename(columns=re_cols['thr'], inplace=True)


# CHOOSE KEY
key = "Material"

# SELECT COLUMNS
cols_exp = ['Source DOI', key, density_exp, 
            'Deposition Temperature (°C)', 'Tpostdep', 'PEALD?', 'Density Meas',
            'k', 'thickness of meas. k (nm)', 'RF Bias (V)', 
            # 'P1 (metal)', 'P2 (nonmetal)', 'P3 (optional)', 'P4',
            'Paper also measured']
df_exp = df_exp[cols_exp]

cols_thr = [key, 'Phase', density_thr, 'Space group']
df_thr = df_thr[cols_thr]


# MERGE DATAFRAMES
cols = ([density_exp], [density_thr, "Phase"])
df_merged = ald_merge_df(df_exp, df_thr, key, cols, view_data=False)



# SORT VALUES
def sort_valuecounts(df, key):
    df = df.loc[:] # prevent SettingWithCopyWarning
    df['freq'] = df.groupby(key)[key].transform('count')
    df.sort_values(['freq',key], ascending=False, inplace=True)
    df.drop('freq', axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


df_exp = sort_valuecounts(df_exp, key)
df_thr = sort_valuecounts(df_thr, key)
df_merged = sort_valuecounts(df_merged, key)


# PRINT DF INFO
ald_print_info(df_exp, label="DF EXPERIMENTAL", vc=[False, key])
ald_print_info(df_thr, label="DF THEORETICAL", vc=[False, key])
ald_print_info(df_merged, label="DF MERGED", vc=[False, key], head=[False, 0], tail=[True, 10])


def create_latex_labels(label):
    if type(label) is float and np.isnan(label):
        return label

    label = str(label)
    lbls = label.split(" ", 1)
    new_label = []
    for lbl in lbls:
        lbl = str(lbl)
        nlbl =  re.sub(r'([a-zA-Z])(\d+)', r'\1_{\2}',lbl)
        nlbl = lbl if (lbl==nlbl) else f"${nlbl}$"
        new_label.append(nlbl)

    label = " ".join(new_label)
    return label


# print(create_latex_labels("NiO"))

# PLOT

def plot_data(df, x, y, z):
    print("PLOT", "-"*60)

    fig, ax = plt.subplots(figsize=(16,9)) # figsize=(6, 7)
    # ax.grid(True, color = '#e8e8e6', linestyle = '--', linewidth = 0.5)
    # ax = fig.add_subplot()
    # ax = sns.lineplot(data=df, x=x, y=y, hue=z, marker="o", ci=None, markersize=4, alpha = 0.9, linestyle='')
    # sns.lineplot(data=df, x=x, y=y, ax=ax, sort=False, hue=z, dashes=False, alpha=0.3, legend=False, ci=None)
    order_z = list(df.sort_values(by=[z], ascending=True)[z].unique())
    # random.Random(2).shuffle(order_z)
    # print(order_z)

    # g2 = sns.lineplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.4, lw=1, ci=None, legend=False, hue_order=order_z)
    ax.axline((0,0), slope=1, color='silver', lw=1, ls='--', alpha=0.5, label='_none_')
    g1 = sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.6, s=35, style=z, hue_order=order_z, style_order=order_z)
    g2 = sns.scatterplot(ax=ax, data=df, x=x, y=x, color="black", alpha=0.9, s=15, marker="+")

    # filt = True
    cols = [z, x]
    # print(df.head())
    df2 = df.loc[:,[x,y,z]]
    df2.sort_values(by=[x,y], ascending=True, inplace=True)
    df2.drop_duplicates(subset=cols, keep='last', inplace=True)
    # print(df2.head())
    # print(df2.shape)
    for line in range(0,df2.shape[0]):
        ox=0
        oy=0.1
        px=df2[x].iloc[line]
        py=df2[y].iloc[line]
        ty=px
        tx=px

        # if line % 3 == 0:
        #     ox = 0.5
        #     oy = 0.75
        # elif line % 3 == 1:
        #     ox = 0.5
        #     oy = 1.25
        # elif line % 3 == 2:
        #     ox = 0.5
        #     oy = 1.75
        # else:
        #     ox = 1.0
        #     oy = 0

        # if (py < px) and (px-py < 0.5):
        #     ox = -0.5
        #     oy = oy*-1
        #     # py=px

        if py+oy > 12.2: 
            oy = 0.3
            ox = 0.5

        plt.annotate(df2[z].iloc[line], 
            xy=(tx, ty), 
            xytext=(px-ox, py+oy), 
            horizontalalignment='center', verticalalignment='bottom', size=8, color='black', weight='light',
            arrowprops={"arrowstyle":"->, widthA=.5, widthB=.5", "color":"gray", "alpha":0.4})


    #place legend outside top right corner of plot
    # plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


    return fig


def plot_swarm(df, x, y, z):
    fig = plt.figure(figsize=(16,9))

    # fig, ax = plt.subplots(figsize=(10,8)) # figsize=(6, 7)

    order_z = list(df.sort_values(by=[z], ascending=True)[z].unique())
    g1 = sns.scatterplot(data=df, x=y, y=y, ci=None, hue=x, alpha=0.6, s=35, style=z, style_order=order_z)



    # g1 = sns.catplot(data=df, x=y, y=y, hue=x, col=z, col_wrap=7, alpha=0.8, sharex=False, sharey=False, kind="strip", jitter=False, height=5, aspect=1.25, ci=None)
    # # plt.xticks(rotation=90)

    # # ax1 = g1.axes[0]

    # # g1.set(ylim=(0,None), xlim=(0,None))

    # for ax in g1.axes.flatten():
    #     ax.axline((0,0), slope=1, color='silver', lw=1, ls='--', label='_none_')
    #     ax.tick_params(labelbottom=True, labelleft=True)

    #     ax.set(xlim=(0,12))  # x axis starts at 0
    #     ax.set(ylim=(0,12))  # y axis starts at 0




    return fig

def plot_data1(df, x, y, z, point_labels,  **kwargs):
    ald_print_info(df, label="DF PLOTTING")

    fig, ax = plt.subplots(figsize=(10,10)) # figsize=(6, 7)

    order_z = list(df[z].unique())

    ax.axline((0,0), slope=1, color='silver', lw=1, ls='--', alpha=0.5, label='_none_')
    g1 = sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.6, s=35, style=z, hue_order=order_z, style_order=order_z)
    g2 = sns.scatterplot(ax=ax, data=df, x=x, y=x, color="black", alpha=0.9, s=15, marker="+")

    plt.minorticks_on()

    # SET TITLE
    plt.title("ALD Thin Film Density")

    # SET AXIS LABELS
    ax.set_xlabel(" ".join([x, kwargs.get('units')]))
    ax.set_ylabel(" ".join([y, kwargs.get('units')]))

    df_lbls = df.loc[:, [x, y, *point_labels]]
    df_lbls.sort_values(by=[x,y], ascending=True, inplace=True)
    df_lbls.drop_duplicates(subset=[x, z], keep='last', inplace=True)

    for lbl_col in point_labels:
        # print(df_lbls[lbl_col].head())
        df_lbls[lbl_col] = df_lbls[lbl_col].apply(create_latex_labels)
    
    # df_lbls[point_labels[0]] = df_lbls[point_labels[0]].apply(create_latex_labels)


    print(df_lbls.head())
    print(df_lbls.shape)

    texts = []

    for line in range(0,df_lbls.shape[0]):
        lbls = list(df_lbls.loc[:,point_labels].iloc[line])
        # print(lbls)

        lvls = {"none":0, "base":1.0,"xsm": 0.1, "sm": 1, "md": 2, "lg": 4, "xlg": 5}
        ox=lvls["sm"]
        oy=lvls["base"]


        # CREATE LABEL TEXT
        pnt_lbl = ""
        len_twolines = 0 
        if (type(lbls[1]) is float and np.isnan(lbls[1])):
            # if nan in points_labels set label to material
            pnt_lbl = lbls[0]

        elif (len(lbls) > 1) and (re.sub(r'\$','',lbls[0]) in lbls[1]):
            pnt_lbl = lbls[1]

        else:
            lbls = [lbls[0], *lbls[1].split(" ", 1)]
            len_twolines = len(lbls[1]) 
            pnt_lbl = "\n".join(lbls)

        nlines = pnt_lbl.count("\n")+1
        oneline = nlines == 1
        twolines = nlines == 2
        manylines = nlines >= 3

        lg_twolines = len_twolines > 6

        # CREATE SECTIONS FOR LABEL PLACEMENT
        mx = 0.055 # 5% margins
        my = 0.03 # 3% margins
        delta_x = ax.get_xlim()[1] - ax.get_xlim()[0]
        delta_y = ax.get_ylim()[1] - ax.get_ylim()[0]

        x_margin = delta_x*mx
        y_margin = delta_y*my

        ylim_l = ax.get_ylim()[0] + (y_margin)
        ylim_h = ax.get_ylim()[1] - (y_margin)

        xlim_l = ax.get_xlim()[0] + (x_margin)
        xlim_h = ax.get_xlim()[1] - (x_margin)
        
        sec_x1 = delta_x / 3.0
        sec_x2 = ax.get_xlim()[1] - sec_x1

        # print("sections ", sec_x1, sec_x2)

        x_thr=df_lbls[x].iloc[line]
        y_thr=df_lbls[x].iloc[line]
        y_exp=df_lbls[y].iloc[line]


        below_line = y_thr > y_exp
        # if y_thr > y_exp: 
        #     y_exp = y_thr

        point_in_section_x1 = x_thr < sec_x1
        point_in_section_x3 = x_thr > sec_x2
        point_in_section_x2 = not(point_in_section_x1 or point_in_section_x3)

        point_in_margin_xl = x_thr <= xlim_l
        point_in_margin_xh = x_thr >= xlim_h

        point_in_margin_yl = y_exp <= ylim_l
        point_in_margin_yh = y_exp >= ylim_h

        y_txt = y_exp
        x_txt = x_thr

        # ADJUST TEXT PLACEMENT OFFSETS
        if below_line:
            ox=lvls["xsm"]
            oy=lvls["md"] # increasing oy raises the text on the plot



            if point_in_margin_yl:
                print(f"MARGIN / Y L {pnt_lbl}\n")
                ox = lvls["sm"] * -1 # move text towards right edge of plot
                oy = lvls["xsm"] # move text away from bottom of plot

            elif point_in_section_x1:
                # print(f"SECTION / X1 {pnt_lbl}\n")
                ox = lvls["md"] # larger offset
                oy = lvls["md"] # larger offset
                ox *= 2.5
                oy *= 1.5
                if line % 5 == 0:
                    ox *= 0.5
                    oy *= -1.2

                    if line % 2 == 0:  # DONE
                        ox *= 1.45
                        oy *= 0.65

                    if line % 2 == 1: 
                        ox *= 1.5
                        oy *= 1.2

                    if oneline: # DONE
                        ox *= 0.13
                        oy *= 0.5


                    if manylines:
                        ox *= 2.0

                elif line % 5 == 1: # DONE
                    ox *= 0.2
                    oy *= -0.8


                    if oneline: # DONE
                        ox *= 0.2
                        oy *= 1.3

                    if line % 3 == 0: # DONE
                        ox *= 6.0
                        oy *= 0.05

                    if line % 3 == 1: # DONE
                        ox *= 3.0
                        oy *= 0.6

                    if line % 3 == 2: # DONE
                        ox *= 2.5
                        oy *= 0.4



                elif line % 5 == 2: # DONE
                    ox *= 0.8
                    oy *= -0.5

                    if line % 3 == 0:
                        ox *= 2.0
                        oy *= 3.0
                    if line % 3 == 1:
                        ox *= 1.5
                        oy *= 1.2
                    if line % 3 == 2:
                        ox *= 0.3
                        oy *= 0.2


                    if manylines:
                        ox *= 1.5

                elif line % 5 == 3: # DONE
                    ox *= 1.5
                    oy *= -0.6


                    if line % 2 == 0:
                        ox *= 1.1
                        oy *= 1.2
                    if line % 2 == 1:
                        ox *= 0.8
                        oy *= 0.6


                    if oneline:
                        ox *= 0.2
                        oy *= -0.8

              
                elif line % 5 == 4:
                    ox *= 2.0
                    oy *= -0.4

                    if line % 2 == 0: # DONE
                        ox *= 0.5
                        oy *= 2.9
                    if line % 2 == 1: # DONE
                        ox *= 0.45
                        oy *= 2.4

                    if manylines:
                        ox = lvls["lg"]
                        oy = lvls["lg"] * -1
          

                else:
                    ox *= 1.0
                    oy *= 1.0


            if point_in_margin_xh:
                print(f"MARGIN / X H {pnt_lbl}\n")
                ox = lvls["sm"] * -1 # move text away from right edge of plot

            elif point_in_section_x3:
                # print(f"SECTION / X3 {pnt_lbl}\n")
                ox = lvls["sm"] # larger offset
                ox *= 0.75 # adjust slightly left
                oy *= 1.1



            if point_in_section_x2:
                ox = lvls['md'] * 2.2
                oy *= 1.0

                if line % 5 == 0:
                    ox *= 1.0
                    oy *= 1.0
                if line % 5 == 1:
                    ox *= 2.0
                    oy *= 0.3
                if line % 5 == 2:
                    ox *= 0.7
                    oy *= 0.7
                if line % 5 == 3:
                    ox *= 1.0
                    oy *= 1.0
                if line % 5 == 4:
                    ox *= 1.0
                    oy *= 1.0


                if oneline:
                    ox *= 0.3
                    oy *= 0.4
                if manylines:
                    ox *= 2.0


        else: # above line
            ox=lvls["md"] * -1
            oy=lvls["md"]


            if point_in_margin_xl:
                print(f"MARGIN X L / {pnt_lbl}\n")

            elif point_in_section_x1:
                print(f"SECTION X1 / {pnt_lbl}\n")
                ox *= 1.21
                oy *= 0.7

                if line % 2 == 1:
                    ox *= 1.6
                    oy *= 3.0

                    if line % 4 == 1:
                        ox *= 1.1
                        oy *= 1.75


            if point_in_margin_yh:
                print(f"MARGIN Y H / {pnt_lbl}\n")
                oy *= -1
                oy *= 0.1
            
            elif point_in_section_x3:
                print(f"SECTION X3 / {pnt_lbl}\n")
                if line % 2 == 0:
                    ox *= 2.0
                    oy *= 1.3

                    if line % 4 == 0:
                        ox *= 1.45
                        oy *= 1.25


            if point_in_section_x2:
                if line % 2 == 1:
                    ox *= 3.0
                    oy *= 2.5

                    if line % 4 == 1:
                        oy *= 0.8

                else:
                    ox *= 0.8
                    oy *= 2.0                        


            if not(point_in_section_x2):
                oy *= 0.65
                ox *= 0.75



            if manylines:
                ox=lvls["md"] * -1.2
                oy=lvls["xlg"] * 2.1

                if line % 3 == 0:
                    ox *= 2.0
                    oy *= 1.3
                elif line % 3 == 1:
                    ox *= 0.8
                    oy *= 1.1
                elif line % 3 == 2:
                    ox *= 1.7
                    oy *= 1.0

            elif lg_twolines:
                ox=lvls["md"] * -1.1
                oy=lvls["xlg"] * 1.8


                if line % 3 == 0:
                    ox *= 1.3
                    oy *= 0.4
                elif line % 3 == 1:
                    ox *= 1.0
                    oy *= 0.8
                elif line % 3 == 2:
                    ox *= 0.55
                    oy *= 0.55

                    if line % 2 == 0:
                        ox *= 1.8
                        oy *= 0.8
                    if line % 2 == 1:
                        ox *= 1.3
                        oy *= 1.0


            elif twolines and not(lg_twolines):
                ox=lvls["sm"] * -1.1
                oy=lvls["md"] * 1


                if line % 3 == 0:
                    ox *= 1.2
                    oy *= 1.1
                elif line % 3 == 1:
                    ox *= 1.0
                    oy *= 1.3
                elif line % 3 == 2:
                    ox *= 1.4
                    oy *= 1.6

                    if line % 2 == 0:
                        ox *= 0.8
                        oy *= 1.3
                    elif line % 2 == 1:
                        ox *= 0.8
                        oy *= 0.8



            elif oneline:
                ox=lvls["sm"] * -1
                oy=lvls["sm"]


                if line % 3 == 0:
                    ox *= 1.5
                    oy *= 1.6
                elif line % 3 == 1:
                    ox *= 1.5
                    oy *= 0.8
                elif line % 3 == 2:
                    ox *= 1.4
                    oy *= 1.0



        y_txt += oy
        x_txt += ox


        # BRING ANNOTATIONS INSIDE THE MARGINS OF THE PLOT
        while y_txt > ylim_h:
            # print(f"\nH {y_exp+oy:.4f} > ylim_h {ylim_h:.4f} -- {oy}")
            if abs(round(y_txt, 5)) <= abs(round(ylim_h, 5)):
                y_txt = ylim_h

            if xlim_h < 0:
                y_txt *= 1.01
            else:
                y_txt *= 0.99

        while x_txt > xlim_h:
            # print(f"\nH {x_txt:.4f} > xlim_h {xlim_h:.4f} -- {ox}")
            if abs(round(x_txt, 5)) <= abs(round(xlim_h, 5)):
                x_txt = xlim_h

            if xlim_h < 0:
                x_txt *= 1.01
            else:
                x_txt *= 0.99

        while y_txt < ylim_l:
            # print(f"\nL {y_txt:.4f} < ylim_l {ylim_l:.4f}")
            if abs(round(y_txt, 5)) >= abs(round(ylim_l, 5)):
                y_txt = ylim_l

            if ylim_l < 0:
                y_txt *= 0.995
            else:
                y_txt *= 1.005

        while x_txt < xlim_l:
            # print(f"\nL {x_thr-ox:.4f} < xlim_l {xlim_l:.4f}")
            if abs(round(x_txt, 5)) >= abs(round(xlim_l, 5)):
                x_txt = xlim_l

            if xlim_l < 0:
                x_txt *= 0.995
            else:
                x_txt *= 1.005



        pnt_coords = (x_thr, y_exp)
        text_coords = (x_txt, y_txt)
        rad = 0.1 if below_line else -0.1

        texts.append(plt.annotate(pnt_lbl, 
                    xy=pnt_coords, xytext=text_coords, 
                    # verticalalignment='center', horizontalalignment='right', 
                    size=7, color='black', weight='light',
                    arrowprops={"arrowstyle":"-|>, widthA=.4, widthB=.4",
                                "connectionstyle":f"arc3,rad={rad}",
                                "color":"gray", "alpha":0.3
                    })
        )


    # adjust_text(texts, precision=0.001,
    #     force_text=(0.01, 0.1), 
    #     force_points=(0.01, 0.1),
    #     expand_text=(1.01, 1.01), 
    #     expand_points=(1.01, 1.01),
    #     autoalign='y', avoid_points=False, avoid_self=False,
    #     only_move={'points':'', 'text':'xy', 'objects':''})
    
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    return fig




# # df_merged.plot(x=density_thr, y=density_exp, marker='x', kind='scatter') # scatter plot
# fig1 = plot_data(df_merged, x=density_thr, y=density_exp, z="Material")
fig2 = plot_data1(df_merged, x=density_thr, y=density_exp, z=key, point_labels=[key, "Phase"], units=density_units)
# fig2 = plot_swarm(df_tidy, x="Type", y="Density", z="Material")


# fig1.savefig('plots/plotDensities.png', dpi=200, bbox_inches="tight")
fig2.savefig('plots/plotDensities-4.png', dpi=200, bbox_inches="tight")

# plt.show()

