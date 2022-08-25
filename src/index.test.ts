import { describe, expect, it } from '@jest/globals';
import { parseSchema, transposeRows, parseOutput } from './index'

describe("transposeRows", () => {
    it("returns transposes correctly", () => {
        const rows = [
            { "hello": 3, "world": 4 },
            { "hello": 4, "world": 5 },
        ]
        expect(transposeRows({
            hello: "string", world: "int32"
        }, rows, ["hello", "world"])).toEqual({ hello: [3, 4], world: [4, 5] })
    })
})