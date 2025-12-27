// @ts-check

/** @type {import('@docusaurus/types').PluginModule} */
const CustomFooterPlugin = (context, options) => {
  return {
    name: 'docusaurus-plugin-custom-footer',
    
    getClientModules() {
      return [
        require.resolve('./CustomFooterInjector'),
      ];
    },
  };
};

module.exports = CustomFooterPlugin;