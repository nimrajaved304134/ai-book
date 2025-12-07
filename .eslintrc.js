module.exports = {
  root: true,
  extends: [
    '@docusaurus',
    'prettier',
  ],
  plugins: ['prettier'],
  rules: {
    'prettier/prettier': 'error',
  },
};