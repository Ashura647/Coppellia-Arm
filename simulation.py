from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

def get_handle_by_name(name: str):
    h = sim.getObject(name, {"noError": True})  # accepts alias too
    if h != -1:
        return h

    # fallback: search all objects and match their names
    objs = sim.getObjects(sim.handle_all)
    for o in objs:
        if sim.getObjectName(o) == name:
            return o
    return -1

table = get_handle_by_name('Table')      # try 'Table', 'table', 'diningTable', etc.
arm   = get_handle_by_name('P-ARM')      # or whatever your arm is named

if table == -1:
    raise RuntimeError("Couldn't find an object named 'Table'. Check the scene tree name.")
if arm == -1:
    raise RuntimeError("Couldn't find an object named 'P-ARM'. Check the scene tree name.")

# Put arm on top of table
table_pos = sim.getObjectPosition(table, -1)
table_bb  = sim.getObjectFloatParam(table, sim.objfloatparam_objbbox_max_z)  # top in local bbox
arm_bbmin = sim.getObjectFloatParam(arm,   sim.objfloatparam_objbbox_min_z)

# safest: just set Z a bit above table top
z_on_table = table_pos[2] + 0.05  # tweak if needed
sim.setObjectPosition(arm, -1, [table_pos[0], table_pos[1], z_on_table])

# Optional: parent the arm to the table
sim.setObjectParent(arm, table, True)

print("Arm placed on table.")
