const LevelSchema = {
    type: "object",
    properties: {
        title: { type: "string" },
        intervals: {
            type: "array",
            items: {
                type: "array",
                items: { type: "integer" }
            }
        }
    },
    required: ["title", "intervals"],
    additionalProperties: false
};
export { LevelSchema };
