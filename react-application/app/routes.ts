import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
    index("skeleton-viewer/PoseComparisonPage.tsx"),
    route("level-creation", "level-creation/LevelCreationPage.tsx")
] satisfies RouteConfig;
