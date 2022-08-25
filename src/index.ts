// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
const ort = require('onnxruntime-node')
import schema from '../schema.json'
//const ort = require();
//const schema = require('./schema.json')

interface Dictionary<T> {
    [Key: string]: T;
}
//this converts [{key:value}] into {key:[value]}
export const transposeRows = (arr: Dictionary<any>[], keys: string[]) => {
    return arr.reduce((aggr, row) => {
        for (const key of keys) {
            aggr[key].push(row[key])
        }
        return aggr
    }, keys.reduce((aggr, key) => ({
        ...aggr,
        [key]: []
    }), {}))
}
export const parseSchema = (schema: Dictionary<any>, arr: Dictionary<any>[]) => {
    const keys = Object.keys(schema)
    const transposedRows = transposeRows(arr, keys)
    const n = arr.length
    return keys.reduce((aggr, key) => {
        return { ...aggr, [key]: new ort.Tensor(schema[key], transposedRows[key], [n, 1]) }
    }, {})
}

interface LabelOutput {
    dims: number[],
    type: string,
    size: number,
    data: any[]
}
interface ProbabilityOutput {
    dims: number[],
    type: string,
    size: number,
    data: Float32Array
}
interface ModelOutput {
    label: LabelOutput,
    probabilities: ProbabilityOutput
}
interface Predictions {
    value: any,
    probability: number
}
export const parseOutput = (results: ModelOutput) => {
    const numLabels = results.label.dims[0]
    const parsedResults: Predictions[] = []
    results.label.data.forEach((v, index) => {
        const value = results.label.type === 'int64' ? parseInt(v.toString()) : v
        let maxProbability = 0
        for (let i = index * numLabels; i < (index + 1) * numLabels; ++i) {
            if (maxProbability < results.probabilities.data[i]) {
                maxProbability = results.probabilities.data[i]
            }
        }
        const result = {
            value, probability: maxProbability
        }
        parsedResults.push(result)
    })
    return parsedResults
}


exports.handler = async function (event: any, context: any) {
    const session = await ort.InferenceSession.create('./pipeline_titanic.onnx');
    const feeds = parseSchema(schema, event.body)
    const results = await session.run(feeds);
    return parseOutput(results)
    //console.log("EVENT: \n" + JSON.stringify(event, null, 2))
    //return context.logStreamName
    //event
}

// use an async context to call onnxruntime functions.
async function main() {
    try {
        const session = await ort.InferenceSession.create('./pipeline_titanic.onnx');

        const feeds = parseSchema(schema, [{
            sex: "female",
            age: 20.0,
            fare: 3.5,
            embarked: "S",
            pclass: BigInt(1)
        },
        {
            sex: "male",
            age: 25.0,
            fare: 6.0,
            embarked: "S",
            pclass: BigInt(2)
        }])

        const results = await session.run(feeds);
        console.log(parseOutput(results))

    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

//main();