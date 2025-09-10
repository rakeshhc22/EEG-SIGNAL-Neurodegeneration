import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function Chart() {
  const data = {
    labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
    datasets: [
      {
        label: "Seizures Detected",
        data: [3, 2, 5, 4, 6, 3, 7],
        borderColor: "#2b6cb0",
        backgroundColor: "#2b6cb0",
        tension: 0.3,
        fill: false,
      },
      {
        label: "Processed Signals",
        data: [50, 65, 70, 80, 95, 110, 120],
        borderColor: "#38a169",
        backgroundColor: "#38a169",
        tension: 0.3,
        fill: false,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: { display: true, text: "EEG Analysis Trends" },
    },
  };

  return <Line data={data} options={options} />;
}
