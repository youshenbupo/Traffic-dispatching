"""Generate SUMO grid network scenarios and traffic demand."""
import os
import sys
import argparse
import subprocess
import random
from xml.etree import ElementTree as ET

try:
    import traci
    import sumolib
except ImportError:
    traci = None
    sumolib = None


def generate_grid(rows: int, cols: int, out_dir: str, edge_length: float = 200.0,
                  num_lanes: int = 1, speed_limit: float = 13.89,
                  demand: float = 1000.0, seed: int = 42):
    """Generate a rows x cols SUMO grid network with random traffic demand.

    Args:
        rows: number of intersections vertically
        cols: number of intersections horizontally
        out_dir: output directory for the scenario
        edge_length: length of each edge in meters
        num_lanes: number of lanes per edge
        speed_limit: max speed in m/s
        demand: total vehicles per hour (approx)
        seed: random seed
    """
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    net_file = os.path.join(out_dir, "grid.net.xml")
    route_file = os.path.join(out_dir, "grid.rou.xml")
    cfg_file = os.path.join(out_dir, "grid.sumocfg")

    # Use netgenerate to create a grid network
    # Build list of grid intersection ids (letters for columns, numbers for rows)
    cols_letters = [chr(ord("A") + c) for c in range(cols)]
    tls_ids = [f"{col}{row}" for col in cols_letters for row in range(rows)]

    cmd = [
        "netgenerate",
        "--grid", "true",
        "--grid.number", str(rows),
        "--grid.length", str(edge_length),
        "--grid.attach-length", str(edge_length),
        "--no-internal-links", "false",
        "--default.lanenumber", str(num_lanes),
        "--default.speed", str(speed_limit),
        "--tls.set", ",".join(tls_ids),
        "--tls.default-type", "static",
        "--tls.green.time", "27",
        "--tls.yellow.time", "3",
        "--output-file", net_file,
        "--seed", str(seed),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Add traffic lights manually if not present
    _ensure_traffic_lights(net_file)

    # Generate random trips
    period = 3600.0 / max(1.0, demand)
    trip_file = os.path.join(out_dir, "trips.trips.xml")
    # Locate randomTrips.py
    random_trips = os.path.join(os.environ.get("SUMO_HOME", ""), "tools", "randomTrips.py")
    if not os.path.exists(random_trips):
        try:
            import sumo
            random_trips = os.path.join(sumo.SUMO_HOME, "tools", "randomTrips.py")
        except Exception:
            random_trips = ""
    if not os.path.exists(random_trips):
        random_trips = "/usr/share/sumo/tools/randomTrips.py"

    cmd_trips = [
        sys.executable, random_trips,
        "-n", net_file,
        "-o", trip_file,
        "-r", route_file,
        "--period", str(period),
        "--begin", "0",
        "--end", "3600",
        "--seed", str(seed),
        "--validate",
        "--vehicle-class", "passenger",
    ]
    # Handle missing SUMO_HOME tools path gracefully
    if not os.path.exists(random_trips):
        cmd_trips[0] = "randomTrips.py"
        cmd_trips[1] = "-n"
    print("Running:", " ".join(cmd_trips))
    try:
        subprocess.run(cmd_trips, check=True)
    except Exception as e:
        print(f"Warning: randomTrips failed ({e}). Using fallback route generation.")
        _generate_simple_routes(net_file, route_file, demand, seed)

    # Write sumocfg
    _write_sumocfg(cfg_file, net_file, route_file)
    print(f"Scenario generated at {out_dir}")


def _ensure_traffic_lights(net_file: str):
    """Ensure all junctions have traffic lights."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    for junction in root.findall("junction"):
        jtype = junction.get("type")
        if jtype == "traffic_light":
            continue
        if jtype in ["dead_end", "internal", "unregulated"]:
            continue
        # Convert to traffic light
        junction.set("type", "traffic_light")
    tree.write(net_file, encoding="UTF-8", xml_declaration=True)


def _generate_simple_routes(net_file: str, route_file: str, demand: float, seed: int):
    """Fallback route generator using net edges."""
    random.seed(seed)
    tree = ET.parse(net_file)
    root = tree.getroot()
    edges = [e.get("id") for e in root.findall("edge") if ":" not in e.get("id")]
    n_veh = int(demand)
    with open(route_file, "w", encoding="utf-8") as f:
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '\
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>\n')
        for i in range(n_veh):
            depart = random.uniform(0, 3600)
            src = random.choice(edges)
            dst = random.choice(edges)
            while dst == src:
                dst = random.choice(edges)
            f.write(f'    <vehicle id="veh{i}" type="car" depart="{depart:.2f}">\n')
            f.write(f'        <route edges="{src} {dst}"/>\n')
            f.write('    </vehicle>\n')
        f.write('</routes>\n')


def _write_sumocfg(cfg_file: str, net_file: str, route_file: str):
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", value=os.path.basename(net_file))
    ET.SubElement(inp, "route-files", value=os.path.basename(route_file))
    time = ET.SubElement(cfg, "time")
    ET.SubElement(time, "begin", value="0")
    ET.SubElement(time, "end", value="3600")
    ET.SubElement(time, "step-length", value="1")
    tree = ET.ElementTree(cfg)
    tree.write(cfg_file, encoding="UTF-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--edge_length", type=float, default=200.0)
    parser.add_argument("--num_lanes", type=int, default=1)
    parser.add_argument("--speed_limit", type=float, default=13.89)
    parser.add_argument("--demand", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_grid(args.rows, args.cols, args.out_dir, args.edge_length,
                  args.num_lanes, args.speed_limit, args.demand, args.seed)


if __name__ == "__main__":
    main()
