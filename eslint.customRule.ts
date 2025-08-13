import { TSESTree, ESLintUtils } from "@typescript-eslint/utils";
import { RuleContext } from "@typescript-eslint/utils/dist/ts-eslint";

export const rules = {
    "disallow-array-for-object-type": {
        meta: {
            type: "problem",
            docs: {
                description: "Disallow array for object type.",
                recommended: false
            },
            messages: {
                disallowArrayForObjectType: "Array for object type are disallowed."
            },
            schema: []
        },
        create(context: RuleContext<"disallowArrayForObjectType", []>) {
            const parserServices = ESLintUtils.getParserServices(context);

            return {
                TSArrayType(node: TSESTree.TSArrayType) {
                    const tsNode = parserServices.esTreeNodeToTSNodeMap.get(node.elementType);
                    const checker = parserServices.program.getTypeChecker();
                    const type = checker.getTypeAtLocation(tsNode);

                    const indexType = type.getStringIndexType();
                    if (indexType) {
                        context.report({
                            node,
                            messageId: "disallowArrayForObjectType"
                        });
                    }
                }
            };
        }
    }
};
