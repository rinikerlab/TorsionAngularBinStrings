.. _theory:

Standard TABS Methodology
==========================

.. figure:: images/standardTABSmethod.png
   :width: 100%
   :align: center

   Standard TABS methodology overview.

#. **Identification of torsions of interest**:

   For a given molecule, the torsions of interest are defined through a predefined set of SMARTS patterns [SCST13]_ [GWMA16]_ plus any additional non-trivial 
   (atoms involved in the bond have at least one non-hydrogen neighbor) rotatable bonds. 

#. **Retrieving torsion information and fitting of the distributions to identify torsion states**:

   For each identified torsion, experimental-torsinal preferences in form of torsion-angle distributions are retrieved. 
   The fits were performed by Riniker and Landrum [RSLG15]_.

#. **Identification of torsion states (space discretization)**

   Based on the fitted distributions, torsion states are identified as local maxima in the distributions separated by local minima.

#. **Assigning labels to conformers based on the torsion states**

   For a given conformer, each torsion is assigned a label corresponding to the torsion state it falls into. 
   The collection of all torsion labels for a conformer constitutes its TABS label.

.. rubric:: References

.. [SCST13] Schärfer, C., Schulz-Gasch, T., Ehrlich, H.-C., Guba, W., Rarey, M., Stahl, M.: Torsion Angle Preferences in Druglike Chemical Space: A Comprehensive Guide. J. Med. Chem. 56, 2016–2028 (2013).
.. [GWMA16] Guba, W., Meyder, A., Rarey, M., Hert, J.: Torsion Library Reloaded: A New Version of Expert-Derived SMARTS Rules for Assessing Conformations of Small Molecules. J. Chem. Inf. Model. 56, 1–5 (2016)
.. [RSLG15] Riniker, S., Landrum, G. A.: Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation. J. Chem. Inf. Model. 55, 2562–2574 (2015).