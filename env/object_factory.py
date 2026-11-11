import math
from typing import Any, Dict

import pymunk


def _apply_common_shape_properties(shape, config: Dict[str, Any]):
    shape.friction = float(config.get("friction", 0.8))
    shape.elasticity = float(config.get("elasticity", 0.2))


def _apply_common_body_properties(body, config: Dict[str, Any]):
    body.position = float(config.get("x", 0.0)), float(config.get("y", 0.0))
    body.angle = math.radians(float(config.get("angle_degrees", 0.0)))

    velocity = config.get("velocity", {"x": 0.0, "y": 0.0})
    body.velocity = float(velocity.get("x", 0.0)), float(velocity.get("y", 0.0))

    body.angular_velocity = float(config.get("angular_velocity", 0.0))


def _make_body(space, config: Dict[str, Any], mass: float, moment: float):
    dynamic = bool(config.get("dynamic", True))

    if dynamic:
        body = pymunk.Body(mass, moment)
    else:
        body = space.static_body

    return body


def create_circle(space, config: Dict[str, Any]):
    radius = float(config["radius"])
    mass = float(config.get("mass", 1.0))

    dynamic = bool(config.get("dynamic", True))

    if dynamic:
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        _apply_common_body_properties(body, config)
    else:
        body = space.static_body

    shape = pymunk.Circle(body, radius)

    if not dynamic:
        shape.body = space.static_body
        shape.offset = (float(config.get("x", 0.0)), float(config.get("y", 0.0)))

    _apply_common_shape_properties(shape, config)

    if dynamic:
        space.add(body, shape)
    else:
        space.add(shape)

    return {
        "id": config["id"],
        "type": "circle",
        "body": body,
        "shape": shape,
        "config": config,
    }


def create_box(space, config: Dict[str, Any]):
    width = float(config["width"])
    height = float(config["height"])
    mass = float(config.get("mass", 1.0))

    dynamic = bool(config.get("dynamic", True))

    if dynamic:
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        _apply_common_body_properties(body, config)
    else:
        body = space.static_body

    shape = pymunk.Poly.create_box(body, (width, height))

    if not dynamic:
        shape.body = space.static_body
        # For static box, vertices are around origin, so we move using transform.
        x = float(config.get("x", 0.0))
        y = float(config.get("y", 0.0))
        angle = math.radians(float(config.get("angle_degrees", 0.0)))
        transform = pymunk.Transform.translation(x, y) @ pymunk.Transform.rotation(angle)
        verts = [transform @ v for v in shape.get_vertices()]
        shape = pymunk.Poly(space.static_body, verts)

    _apply_common_shape_properties(shape, config)

    if dynamic:
        space.add(body, shape)
    else:
        space.add(shape)

    return {
        "id": config["id"],
        "type": "box",
        "body": body,
        "shape": shape,
        "config": config,
    }


def create_polygon(space, config: Dict[str, Any]):
    vertices = config["vertices"]
    local_vertices = [(float(v["x"]), float(v["y"])) for v in vertices]

    mass = float(config.get("mass", 1.0))
    dynamic = bool(config.get("dynamic", True))

    if dynamic:
        moment = pymunk.moment_for_poly(mass, local_vertices)
        body = pymunk.Body(mass, moment)
        _apply_common_body_properties(body, config)
        shape = pymunk.Poly(body, local_vertices)
        _apply_common_shape_properties(shape, config)
        space.add(body, shape)
    else:
        x = float(config.get("x", 0.0))
        y = float(config.get("y", 0.0))
        angle = math.radians(float(config.get("angle_degrees", 0.0)))

        transform = pymunk.Transform.translation(x, y) @ pymunk.Transform.rotation(angle)
        world_vertices = [transform @ pymunk.Vec2d(v[0], v[1]) for v in local_vertices]

        body = space.static_body
        shape = pymunk.Poly(body, world_vertices)
        _apply_common_shape_properties(shape, config)
        space.add(shape)

    return {
        "id": config["id"],
        "type": "polygon",
        "body": body,
        "shape": shape,
        "config": config,
    }


def create_segment(space, config: Dict[str, Any]):
    dynamic = bool(config.get("dynamic", False))

    ax = float(config["a"]["x"])
    ay = float(config["a"]["y"])
    bx = float(config["b"]["x"])
    by = float(config["b"]["y"])
    thickness = float(config.get("thickness", 8.0))

    if dynamic:
        mass = float(config.get("mass", 1.0))

        # Segment body is placed at midpoint, vertices are local.
        mid_x = (ax + bx) / 2.0
        mid_y = (ay + by) / 2.0

        local_a = (ax - mid_x, ay - mid_y)
        local_b = (bx - mid_x, by - mid_y)

        length = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        moment = pymunk.moment_for_segment(mass, local_a, local_b, thickness / 2.0)

        body = pymunk.Body(mass, moment)
        body.position = mid_x, mid_y

        shape = pymunk.Segment(body, local_a, local_b, thickness / 2.0)

        _apply_common_shape_properties(shape, config)
        space.add(body, shape)

    else:
        body = space.static_body
        shape = pymunk.Segment(
            body,
            (ax, ay),
            (bx, by),
            thickness / 2.0,
        )

        _apply_common_shape_properties(shape, config)
        space.add(shape)

    return {
        "id": config.get("id", "segment"),
        "type": "segment",
        "body": body,
        "shape": shape,
        "config": config,
    }


def create_object(space, config: Dict[str, Any]):
    obj_type = config["type"]

    if obj_type == "circle":
        return create_circle(space, config)

    if obj_type == "box":
        return create_box(space, config)

    if obj_type == "polygon":
        return create_polygon(space, config)

    if obj_type == "segment":
        return create_segment(space, config)

    raise ValueError(f"Unsupported object type: {obj_type}")