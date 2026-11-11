import math

import pymunk


class GoalEvaluator:
    def __init__(self, level_data, objects):
        self.level_data = level_data
        self.objects = objects
        self.goal_memory = {}

    def reset(self):
        self.goal_memory = {}

    def check_success(self, update_memory=True):
        goals = self.level_data.get("goals", [])

        if not goals:
            return False

        mode = self.level_data.get("goal_logic", "all")

        # Evaluate every goal before combining the results. Using all()/any()
        # directly on a generator short-circuits and can prevent later
        # stateful goals from updating their hold counters.
        results = [self._check_goal(goal, update_memory) for goal in goals]

        if mode == "any":
            return any(results)

        return all(results)

    def _check_goal(self, goal, update_memory=True):
        goal_type = goal["type"]

        if goal_type == "object_reaches_region":
            return self._object_reaches_region(goal)

        if goal_type == "object_stays_in_region":
            return self._object_stays_in_region(goal, update_memory)

        if goal_type == "object_angle_range":
            return self._object_angle_range(goal)

        if goal_type == "object_above_height":
            return self._object_above_height(goal)

        if goal_type == "object_below_height":
            return self._object_below_height(goal)

        if goal_type == "object_velocity_below":
            return self._object_velocity_below(goal)
        
        if goal_type == "object_stays_above_y":
            return self._object_stays_above_y(goal, update_memory)

        raise ValueError(f"Unsupported goal type: {goal_type}")

    def _get_object_body(self, object_id):
        if object_id not in self.objects:
            raise KeyError(f"Object '{object_id}' not found in objects dictionary.")

        return self.objects[object_id]["body"]

    def _object_reaches_region(self, goal):
        object_id = goal["object_id"]
        region = goal["region"]

        obj = self.objects[object_id]
        shape = obj["shape"]

        if region["shape"] == "circle":
            cx = float(region["x"])
            cy = float(region["y"])
            r = float(region["r"])

            # point_query returns signed distance from the goal centre to the
            # object surface, so this works for circles, boxes, and polygons.
            return float(shape.point_query((cx, cy)).distance) <= r

        if region["shape"] == "rectangle":
            rx = float(region["x"])
            ry = float(region["y"])
            w = float(region["width"])
            h = float(region["height"])

            if isinstance(shape, pymunk.Circle):
                centre = shape.body.local_to_world(shape.offset)
                closest_x = min(max(float(centre.x), rx), rx + w)
                closest_y = min(max(float(centre.y), ry), ry + h)
                return math.hypot(
                    float(centre.x) - closest_x,
                    float(centre.y) - closest_y,
                ) <= float(shape.radius)

            object_points = self._shape_world_vertices(shape)
            region_points = [
                (rx, ry),
                (rx + w, ry),
                (rx + w, ry + h),
                (rx, ry + h),
            ]
            return self._polygons_overlap(object_points, region_points)

        raise ValueError(f"Unsupported region shape: {region['shape']}")

    def _object_stays_in_region(self, goal, update_memory=True):
        goal_id = goal["id"]
        duration_steps = int(goal.get("duration_steps", 60))

        inside = self._object_fully_inside_region(goal)

        if update_memory:
            if inside:
                self.goal_memory[goal_id] = self.goal_memory.get(goal_id, 0) + 1
            else:
                self.goal_memory[goal_id] = 0

        return inside and self.goal_memory.get(goal_id, 0) >= duration_steps

    def _object_fully_inside_region(self, goal):
        obj = self.objects[goal["object_id"]]
        shape = obj["shape"]
        region = goal["region"]

        if obj["type"] == "circle":
            centre = shape.body.local_to_world(shape.offset)
            radius = float(shape.radius)
            points = [(float(centre.x), float(centre.y))]
        else:
            radius = 0.0
            points = self._shape_world_vertices(shape)

        if region["shape"] == "circle":
            cx = float(region["x"])
            cy = float(region["y"])
            goal_radius = float(region["r"])
            return all(
                math.hypot(x - cx, y - cy) + radius <= goal_radius
                for x, y in points
            )

        if region["shape"] == "rectangle":
            rx = float(region["x"])
            ry = float(region["y"])
            right = rx + float(region["width"])
            bottom = ry + float(region["height"])
            return all(
                rx <= x - radius
                and x + radius <= right
                and ry <= y - radius
                and y + radius <= bottom
                for x, y in points
            )

        raise ValueError(f"Unsupported region shape: {region['shape']}")

    @staticmethod
    def _shape_world_vertices(shape):
        if isinstance(shape, pymunk.Poly):
            return [
                (float(point.x), float(point.y))
                for point in (
                    shape.body.local_to_world(vertex)
                    for vertex in shape.get_vertices()
                )
            ]
        if isinstance(shape, pymunk.Segment):
            a = shape.body.local_to_world(shape.a)
            b = shape.body.local_to_world(shape.b)
            return [(float(a.x), float(a.y)), (float(b.x), float(b.y))]
        centre = shape.body.position
        return [(float(centre.x), float(centre.y))]

    @classmethod
    def _polygons_overlap(cls, first, second):
        if not first or not second:
            return False

        first_edges = list(zip(first, first[1:] + first[:1]))
        second_edges = list(zip(second, second[1:] + second[:1]))
        if any(
            cls._segments_intersect(a, b, c, d)
            for a, b in first_edges
            for c, d in second_edges
        ):
            return True
        return cls._point_in_polygon(first[0], second) or cls._point_in_polygon(
            second[0], first
        )

    @staticmethod
    def _point_in_polygon(point, polygon):
        x, y = point
        inside = False
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = previous
            x2, y2 = current
            if (y1 > y) != (y2 > y):
                intersect_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x <= intersect_x:
                    inside = not inside
            previous = current
        return inside

    @staticmethod
    def _segments_intersect(a, b, c, d):
        def orientation(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (
                q[1] - p[1]
            ) * (r[0] - p[0])

        def on_segment(p, q, r):
            return (
                min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
                and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
            )

        o1 = orientation(a, b, c)
        o2 = orientation(a, b, d)
        o3 = orientation(c, d, a)
        o4 = orientation(c, d, b)
        if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
            return True
        if abs(o1) <= 1e-9 and on_segment(a, c, b):
            return True
        if abs(o2) <= 1e-9 and on_segment(a, d, b):
            return True
        if abs(o3) <= 1e-9 and on_segment(c, a, d):
            return True
        if abs(o4) <= 1e-9 and on_segment(c, b, d):
            return True
        return False

    def _object_angle_range(self, goal):
        object_id = goal["object_id"]
        min_angle = float(goal["min_angle"])
        max_angle = float(goal["max_angle"])

        body = self._get_object_body(object_id)
        angle = float(body.angle)

        return min_angle <= angle <= max_angle

    def _object_above_height(self, goal):
        object_id = goal["object_id"]
        y_threshold = float(goal["y"])

        obj = self.objects[object_id]

        # In pygame-style coordinates, smaller y means higher on screen.
        return float(obj["shape"].bb.bottom) <= y_threshold

    def _object_below_height(self, goal):
        object_id = goal["object_id"]
        y_threshold = float(goal["y"])

        obj = self.objects[object_id]

        # In pygame-style coordinates, larger y means lower on screen.
        return float(obj["shape"].bb.top) >= y_threshold

    def _object_velocity_below(self, goal):
        object_id = goal["object_id"]
        max_speed = float(goal.get("max_speed", 10.0))

        body = self._get_object_body(object_id)
        speed = body.velocity.length

        return speed <= max_speed
    
    def _object_stays_above_y(self, goal, update_memory=True):
        goal_id = goal["id"]
        object_id = goal["object_id"]
        min_y = float(goal["min_y"])
        duration_steps = int(goal.get("duration_steps", 60))

        obj = self.objects[object_id]

        # In pygame-style coordinates:
        # smaller y = higher on screen
        # larger y = lower on screen
        is_above_floor_limit = float(obj["shape"].bb.top) < min_y

        if update_memory:
            if is_above_floor_limit:
                self.goal_memory[goal_id] = self.goal_memory.get(goal_id, 0) + 1
            else:
                self.goal_memory[goal_id] = 0

        return (
            is_above_floor_limit
            and self.goal_memory.get(goal_id, 0) >= duration_steps
        )
