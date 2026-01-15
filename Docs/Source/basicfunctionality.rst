.. _Basic Functionality Reference:

Basic Functionality Reference
==============================

Core Class
-----------

The class object which is at the center of the TABS workflow is :code:`DihedralsInfo`.

.. autoclass:: tabs.torsions.DihedralsInfo
    :members: __init__
    :undoc-members:
    :show-inheritance:

An instance of the class :code:`DihedralsInfo` can be created from an RDKit mol object 
using :code:`DihedralInfoFromTorsionLib`.

.. autofunction:: tabs.DihedralInfoFromTorsionLib

Core Class Methods
--------------------

The following class methods are available for the :code:`DihedralsInfo` class:

.. automethod:: tabs.torsions.DihedralsInfo.GetTABS
.. automethod:: tabs.torsions.DihedralsInfo.GetnTABS
.. automethod:: tabs.torsions.DihedralsInfo.GetConformerTorsions