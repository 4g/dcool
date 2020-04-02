"""
A Datacenter environment simulator
Since CFD is slow, we use balls to simulate heat flow
Color of a component represents how hot it is.

Following components can be added:
0. Balls : Balls are representation of hot/cold air in this system. They can be added by either rack or tiles.
           And absorbed by anyone too.
1. Walls, walls are non-absorbent i.e. they do not absorb any heat
2. Racks, generates balls, absorbs balls. Generated balls have same color as rack. Loses heat as balls go out.
3. Tiles, generates balls, absorbs nothing
4. AHU, Absorbs balls, uses energy to convert balls from hot to cold

=======================
^^^^^^^^^^^^^^^^^^^^^^=
^   ----  ----  ---- ^=
^     o     o    o   ^=
^   ----  ----  ---- ^=
^                    ^=
^   ----  ----  ---- ^=
^     o     o    o   ^=
^   ----  ----  ---- ^=
^                    ^=
^^^^^^^^^^^^^^^^^^^^^^=
=======================

o: Cold Tile
^: AHU
----: Rack
"=": wall

"""

