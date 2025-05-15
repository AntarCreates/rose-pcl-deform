# rose-pcl-deform

Here's a simple and clear `README.md` for your project:

---

````markdown
# Voxel Terrain Simulation

This project simulates a rigid ball rolling across a deformable voxel terrain with varying ground stiffness. It visually demonstrates how terrain deformation and ball motion are affected by the underlying stiffness zones, using 3D animation with `matplotlib`.

## Features

- 3D voxel grid terrain with 4 stiffness zones (Very Soft → Very Hard)
- Rigid body (ball) physics including collision, bounce, rolling, and deformation
- Real-time terrain deformation and voxel dislodging
- Visualization with live animation in a `matplotlib` 3D plot

## Requirements

Install the Python dependencies using:

```bash
pip install -r requirements.txt
````

> Make sure you have a GUI backend for `matplotlib` (e.g., `TkAgg` is used by default).

## How to Run

Run the simulation script:

```bash
python multi_terrain_test.py
```

This will open an interactive 3D animation window where you can:

* Rotate the view with your mouse
* Scroll to zoom
* Watch the ball interact differently with each ground stiffness zone

## File Overview

* `multi_terrain_test.py`: Main simulation and visualization script
* `point_test.py`: Uses points instead of surface
* `requirements.txt`: List of required Python packages


