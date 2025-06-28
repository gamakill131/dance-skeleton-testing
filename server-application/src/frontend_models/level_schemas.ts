import ajv, { JSONSchemaType } from "ajv"

interface FrontendLevel {
    title: string,
    intervals: number[][]
}

const LevelSchema: JSONSchemaType<FrontendLevel> = {
    type: "object",
    properties: {
        title: {type: "string"},
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
}

export { LevelSchema }