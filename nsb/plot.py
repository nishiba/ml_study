import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nsb
import itertools


def all_scatter_plot(df, target_name):
    data = nsb.feature.select_numeric(df)
    names = nsb.data_util.columns(data)
    names.remove(target_name)
    n = len(names) * (len(names) - 1) // 2
    f, axarr = plt.subplots(n, 1)
    for (x, y), a in zip(itertools.combinations(names, 2), axarr):
        a.scatter(data[x], data[y], c=data[target_name])
        a.set_xlabel(x)
        a.set_ylabel(y)
        print(x, y)

    f.set_size_inches(7, 6.5 * n)


def scatter_with_boundary(x0, x1, y, model, output=None):
    n = 200
    axis0 = np.linspace(x0.min(), x0.max(), n)
    axis1 = np.linspace(x1.min(), x1.max(), n)
    a, b = np.meshgrid(axis0, axis1)
    z = model.predict(np.c_[a.ravel(), b.ravel()]).reshape(a.shape)
    plt.clf()
    plt.contourf(axis0, axis1, z, 1, alpha=0.2)
    plt.scatter(x0, x1, c=y)
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv('../real_world_machine_learning/event_recommendations/data/merged_train_counts.csv')
    names = ['#friends', '#event_attendees_yes', '#event_attendees_maybe', '#event_attendees_invited',
             '#event_attendees_no', '#friends_attendees_yes', '#friends_attendees_maybe', '#friends_attendees_invited',
             '#friends_attendees_no', 'invited_friend_rate', 'attendee_friend_rate', 'timestamp_to_start_time',
             'joinedAt_to_start_time', 'joinedAt_to_timestamp', 'interested']
    # names = ['#friends', '#event_attendees_yes', '#event_attendees_maybe', 'interested']

    nsb.feature.add_new_column(data, '(div, #friends_attendees_invited, #friends) -> invited_friend_rate')
    nsb.feature.add_new_column(data, '(div, #friends_attendees_yes, #friends) -> attendee_friend_rate')
    nsb.feature.add_new_column(data, '(dt_seconds, start_time, timestamp) -> timestamp_to_start_time')
    nsb.feature.add_new_column(data, '(dt_seconds, start_time, joinedAt) -> joinedAt_to_start_time')
    nsb.feature.add_new_column(data, '(dt_seconds, timestamp, joinedAt) -> joinedAt_to_timestamp')

    all_scatter_plot(data[names], 'interested')
