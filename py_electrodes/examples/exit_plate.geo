SetFactory("OpenCASCADE");
Geometry.NumSubEdges = 100; // nicer display of curve
Mesh.CharacteristicLengthMax = 0.005;
        // Create Plate 
Cylinder(1) = { 0, 0, 1.4565463629470001, 0, 0, 0.02, 0.1, 2 * Pi };
Cylinder(2) = { 0, 0, 1.4555463629470002, 0, 0, 0.022, 0.01, 2 * Pi };
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }

s() = Surface "*";
Physical Surface(100) = { s() };

ReverseMesh Surface { s() };
            