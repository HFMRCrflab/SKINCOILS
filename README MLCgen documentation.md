# Documentation MLCgen Macro for FreeCAD

## Contents
1. General Functionality
2. FreeCAD and Macro Usage
3. User Parameters
4. Functions and Classes
5. Description/Nomenclature of Variables
6. Examples of MLC Shapes

----
## 1. General Functionality

**MLCgen** is a macro for FreeCAD that draws **MLC (Multi Loop Coil)** layers as polygonal wires. The overall shape of each layer consists of several disconnected two-dimensional **sub-shapes**, each following the same pattern:

- Two consecutive sub-shapes are separated by a **gap**.
- Each sub-shape alternates between first a **circular part** and then a **linear part**, ending again on an additional circular one.
- A sub-shape further has an **outer (O)** and **inner (I)** edge, connected at the rear end of the conductor by a **connecting line** (also called **turnaround-line** in the following), and at the front end, FreeCAD automatically connects them by closing the polygon.
- When loosely viewed as a whole, the shape of an individual layer forms a sequence of loops, connected by straight elements and including **gaps** to connect various electronics.
- For multiple layers, the shapes can be generated one after the other within the same macro.

First, user parameters and helper variables are determined to generate a **"to-do list"**, where each element represents either an **arc** or a **line**.
This list is processed recursively — first for the **outer edge**, then, when reaching the end of the list, a **turnaround-line** connects the inner and outer edges, and the **inner edge** is drawn in reverse order via recursion.

There is a distinction between **virtual** and **real reference points**:
- *Virtual points* move along helper lines that go directly from circle center to circle center.
- *Real points* correspond to the actual drawn shape and thus deviate slightly from the virtual ones based on the parameters.

----
## 2. FreeCAD and Using the Macro

**Installation:**
- FreeCAD: [https://www.freecad.org/](https://www.freecad.org/)
- Python: no further steps needed.

**Importing the Macro:**
- In FreeCAD: `Macro` → `Macros...` → set “User macro location” to the `.FCMacro` file path.
- The macro will appear in the list and can be executed or edited there.

**Export:**
- `Ctrl + A` → select all wires to export
- `Ctrl + E` (or File → Export)
- Choose file name and destination.

----
## 3. User Parameters

| Parameter                       | Description                                                                                                                                    | Default                |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| **mm, cm, m, um, degrees**      | SI-unit conversions used for readability and ease of adapting to the base unit used in FreeCAD (usually numbers are interpreted as mm)         | 1, 10, 1000, 1/1000, 1 |
| **layerOffset**                 | Distance between layers along the Z-axis (MLC’s Z, not the scanner’s. The output axis interpretation can be toggled at a later point.          | 400 × um               |
| **X0, Y0, ThetaD0**             | Starting coordinates and angle (°). The point lies on the outer radius, centered within the gap.                                               | 0 × mm, 0 × mm, 0      |
| **Z0**                          | Z offset of the current layer.                                                                                                                 | 0 × layerOffset        |
| **distanceBetweenConductors**   | Distance between the inner edges of the connection lines.                                                                                      | 0.5 × mm               |
| **conductorWidth**              | Width of the conductor, = distance between inner and outer edges.                                                                              | 1.5 × mm               |
| **overlap**                     | Geometric parameter defining the ratio between the distance between the centerpoints of overlapping loops and their outer diameter             | 0.775                  |
| **gapSize**                     | Size of the gaps between two consecutive conducting elements, which is used to place Capacitors or other circuitry                             | 4 × mm                 |
| **arcSampleNumber**             | Number of segments per circular arc. More segments = smoother, but slower. \~300 recommended (\~5 s runtime); >5000 not recommended (\~30 s).  | 300 (or 100)           |
| **relativeSampling** (Boolean)  | If True, `arcSampleNumber` describes a full circle and smaller arcs have less samples accordingly. Else, each arc has the same sample number.  | True (or False)        |
| **flagConstantWidth** (Boolean) | A flag defining whether the width of the MLC is held constant and `loopRadius` and *distanceBetweenCenters* are calculated or vice versa.      | True                   |
| **logging** (Boolean)           | Shows additional log entries (True) or hides them (False).                                                                                     | False                  |
| **totalLength**                 | Total length of the MLC, going from side to side (not apex to apex) of the imaginary hexagon enclosing the MLC.                                | 79 × mm                |
| **loopRadius**                  | Outer radius of each individual loop. By default, this is unused and instead calculated as: `totalLength/(4×overlap×sqrt(3)+2)`                | calculated, 11.5 × mm  |
| **distanceBetweenCenters**      | Distance between centers of adjacent loops of the same layer. By default, this is unused and instead calculated as: `totalLength/2-loopRadius` | calculated, 28 × mm    |
| **lengthOfConnectionLine**      | Distance between outer radii of adjacent loops. By default, this is unused and instead calculated as: `distanceBetweenCenters-2×loopRadius`    | commented out, 1 × mm  |
| **MLCshape** (Matrix of angles) | Defines the MLC’s shape. Each row defines a sub-shape: line breaks between rows represent gaps, commas within rows represent connection lines. | see Chapter 6 examples |

`X0`, `Y0`, `Z0` and `ThethaD0` are defined here as a default starting point, but it is recommended to redefine them along `MLCShape` for each individual shape, as this allows smoother handling through copy-paste.

> ***Note:** Default values are just a working example — they don’t follow PCB design rules or simulation results. Since validity isn’t yet checked, nonsensical physical inputs may lead to faulty output. If this happens, use the defaults as a starting point and adjust reasonably.*

----
## 4. Functions and Classes

**class ConductorElement:**

- Stores key properties of a single element (line or arc). This can easily be expanded for additional shapes.
- notably the attribute `type` defines which type of element one instance portrays, which subsequently indicates which other attributes are relevant. 
- The `toDoList` is a ordered list of these objects portraying directly the order in which the individual elements are drawn.
- The class is essentially used as a tuple, but with improved readability — so getters/setters are willingly omitted and attributes are instead accessed directly.

**Functions:**

- `cosD(X)` → cosine with an input in degrees.
- `sinD(X)` → sine with an input in degrees.
- `getAngleDfromABR(Ax, Ay, Bx, By, Rx, Ry)` → returns the angle between the points A = [`Ax,Ay`] and B = [`Bx`,`By`] around the center point R.
- `log(*arg)` → prints a log entry if `logging` is True.
- `locate(X, Y)` → debugging aid: marks [`X`,`Y`] with an additional 1 µm² square embedded into the shape exactly when the function is called.
- `turn(angleD)` → rotates the reference angle counterclockwise by `angleD` degrees.
- `placePoint(X, Y)` → (alternative, commented out) adds [`X`, `Y`, `Z0`] to point list (outputs each layer along a XY plane).
- `placePoint(X, Z)` → adds [`X`, `Z0`, `Z`] to point list (outputs each layer along a XZ plane).
- `drawPointAtAngle(Rx, Ry, angleD)` → draws one circular segment around center R = [`Rx`, `Ry`] spanning the angle `angleD`.
- `drawArc(OI, Rx, Ry, samples, angleD, offset=0, overlength=0, gapOffsetA=0, gapOffsetB=0)` → draws an arc of `angleD` degrees with `samples` segments around R = [`Rx`, `Ry`]; includes offsets and corrections for gaps. `OI` differenciates between outer and inner arcs.
- `drawLine(length, angleD, offset=0, overlength=0)` → draws a straight line with the length `length`, at an angle `angleD` and while considering `offset` and `overlength`.
- `turnaroundLine(length, angleD)` → connects outer and inner edges with a straight line (with the length `length`, at an angle `angleD`), which is perpendicular to the tangent touching the imaginary full circle at the center of the gap.
- `drawConductorPartOuter(self)` → draws one outer element based on the `type` (arc or line) and other attributes stored in `self`.
- `drawConductorPartInner(self)` → analogous, but for the inner edge.
- `drawConductorRecursively(todoList)` → recursively draws one-by-one outer edges, then the turnaround-line, and finally the inner edges by calling itself recursively in-between outer and inner edges.
- `init()` → resets all internal parameters.
- `drawWholeConductor(todoList)` → prepares and calls the recursive drawing for a single layer. After the function's call, the parameters can be readjusted and another shape can be drawn by calling the function again.

----
## 5. Description/Nomenclature of internal Variables

> ***Note:** “X” and “Y” refer to the internal MLC plane. The `placePoint()` function determines whether the final output corresponds to FreeCAD’s XY or XZ plane.*

| Internal Variables               | Description                                                                           |
|----------------------------------|---------------------------------------------------------------------------------------|
| **angleD**                       | Generic local variable describing an angle in degrees for various functions           | 
| **Ax, Ay**                       | Real reference coordinates                                                            |
| **Bx, By**                       | End coordinates of a circular arc                                                     |
| **correctedLength**              | Adjusted length of turnaroundLine, as the line is not exactly in line with the radius |
| **currentRx, currentRy**         | Current circle center (global backup)                                                 |
| **elementNr, elMax**             | Loop iterator and total number of loops                                               |
| **length**                       | Length of the currently drawn edge                                                    |
| **listAngles**                   | Cumulative angles from listAnglesBase                                                 |
| **listAnglesBase**               | Base list of arc angles per sub-shape (from MLCshape)                                 |
| **listRx, listRy**               | Lists of each loops center coordinates in X and Y respectively                        |
| **samples**                      | Derived number of an individual arc's samples, considers the spanned angle            |
| **shapeCounter, numberOfShapes** | Sub-shape iterator and total number of sub-shapes                                     |
| **OI**                           | Differenciates whether the currently drawn edge is an outer or inner edge             |
| **offset** (and derivatives)     | Various distances differenciating virtual and real reference points                   |
| **overlength** (and derivatives) | Distance offset between a circle and its tangent near their touching point            |
| **points**                       | List of points defining the final wire for FreeCAD                                    |
| **Rx, Ry**                       | Circle center coordinates                                                             |
| **todoList**                     | List of elements to be drawn                                                          |
| **X, Y, ThetaD**                 | Virtual reference coordinates and reference angle                                     |
| **Xtemp, Ytemp, tempD**          | Temporary helper variables                                                            |

----
## 6. Examples of MLC Shapes

### 1. TOP Layer (Hexagonal)
```python
MLCshape = [ #hexagon TOP
    [120,240,150],
    [150,180,30] ,
    [30,120,240] ,
    [120,240,150],
    [150,180,30] ,
    [30,120,240] ]
```

### 2. MID Layer (Triangular)
```python
MLCshape = [ #Triangle MID
    [120,180,120] ,
    [180,150]     ,
    [90,60,180]   ,
    [180,210]     ,
    [90,120,60,30],
    [150,240]     ]
```

### 3. BOT Layer (Triangular)
```python
MLCshape = [ #Triangle BOT
    [180,210]     ,
    [90,120,60,30],
    [150,240]     ,
    [120,180,120] ,
    [180,150]     ,
    [90,60,180]   ]
```
