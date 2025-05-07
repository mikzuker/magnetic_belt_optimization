from parametrization import HalbachRing
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pathlib


class StatisticCollection(object):
    @staticmethod
    def get_heatmap_collection(number_of_magnets, dimensions, polatizations, radius):
        B_amplitudes = []

        for n in number_of_magnets:
            for polarization in polatizations:
                for d in dimensions:
                    magnet_dimensions = [(d[0], d[0], d[0]) for _ in range(n[0])]
                    magnet_polarizations = [
                        (polarization[0], 0, 0) for _ in range(n[0])
                    ]

                    ring = HalbachRing(
                        dimensions=magnet_dimensions,
                        polarizations=magnet_polarizations,
                        radius=radius,
                        num_magnets=n,
                        start_angle=[0],
                        end_angle=[180],
                    )

                    point = np.array([0, 0, 0])
                    B_amplitude = ring.get_field_amplitude_at_point(point)
                    B_amplitudes.append([n[0], polarization[0], d[0], B_amplitude])

        df = pd.DataFrame(
            B_amplitudes,
            columns=["number_of_magnets", "polarization", "dimension", "B_amplitude"],
        )

        number_of_magnets_polarization = (
            "statistics/heatmaps/number_of_magnets_polarization"
        )
        dimensions_polarizations = "statistics/heatmaps/dimensions_polarizations"
        pathlib.Path(number_of_magnets_polarization).mkdir(parents=True, exist_ok=True)
        pathlib.Path(dimensions_polarizations).mkdir(parents=True, exist_ok=True)

        for dim in df["dimension"].unique():
            dim_data = df[df["dimension"] == dim]

            pivot_data = dim_data.pivot(
                index="number_of_magnets", columns="polarization", values="B_amplitude"
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_data, annot=True)
            plt.title(f"Magnetic Field Amplitude (Dimension = {dim}m)")
            plt.xlabel("Polarization (T)")
            plt.ylabel("Number of Magnets")
            plt.savefig(
                f"{number_of_magnets_polarization}/heatmap_dimension_{dim}m.png"
            )
            plt.close()

        for n in df["number_of_magnets"].unique():
            n_data = df[df["number_of_magnets"] == n]

            pivot_data = n_data.pivot(
                index="polarization", columns="dimension", values="B_amplitude"
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_data, annot=True)
            plt.title(f"Magnetic Field Amplitude (Number of Magnets = {n})")
            plt.xlabel("Dimension (m)")
            plt.ylabel("Polarization (T)")
            plt.savefig(f"{dimensions_polarizations}/heatmap_number_of_magnets_{n}.png")
            plt.close()


if __name__ == "__main__":
    number_of_magnets = [[9], [11], [13], [15]]
    dimensions = [[0.02], [0.025], [0.03], [0.035]]
    polatizations = [[-1.6], [-1.4], [-1.2], [-1.0]]
    radius = [0.12]

    StatisticCollection.get_heatmap_collection(
        number_of_magnets, dimensions, polatizations, radius
    )
